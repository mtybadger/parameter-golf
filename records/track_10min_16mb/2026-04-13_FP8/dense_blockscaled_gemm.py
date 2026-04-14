# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Tuple, Type
import math
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

"""
A Hopper FP8 groupwise/blockscaled GEMM example in CuTe DSL.
- Matrix A is MxKxL, row-major / k-major FP8
- Matrix B is NxKxL, row-major / k-major FP8
- Matrix C is MxNxL, row-major("N") or column-major("M")
- Scale tensor SFA is Mxceil_div(K, 128)xL
- Scale tensor SFB is ceil_div(N, 128)xceil_div(K, 128)xL

This matches the scaling behavior of the CUTLASS Hopper grouped/blockscaled FP8
example in `fp8_groupwise_gemm.cu`:
- grouped scale factor along M for A,
- blocked scale factor along K for A,
- blocked scale factor along N and K for B.

The kernel keeps the existing Hopper TMA + WGMMA mainloop, stages SFA through
the same TMA pipeline as A/B, preloads the compact SFB tile into shared memory
once per CTA, and performs the scale fusion in registers after each 128-wide K
group. That is the exact scale-promotion interval used by the CUTLASS example.

Constraints:
* Supported input data types: fp8 (e4m3fn, e5m2)
* Scale tensors must be Float32
* SFA must be physically packed with stride `(1, M, M * ceil_div(K, 128))`;
  use `pack_sfa_tensor_for_hopper(...)` for a standard row-major torch tensor
* A and B must be k-major
* Accumulator type must be Float32
* CTA tile shape must be 128x128 or 128x256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


SCALE_GRANULARITY_M = 1
SCALE_GRANULARITY_N = 128
SCALE_GRANULARITY_K = 128
SCALE_DTYPE = cutlass.Float32


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example of FP8 grouped/blockscaled MxNxKxL GEMM on Hopper."
    )

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(128, 128), (128, 256)],
        default=(128, 128),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.Float8E4M3FN,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.Float8E4M3FN,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=2e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mn) != 2:
        parser.error("--tile_shape_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////


class HopperWgmmaGemmKernel:
    """
    This class implements Hopper FP8 grouped/blockscaled batched matrix
    multiplication with compact scale tensors:
        C = sum_k ((SFA * A)_k * (SFB * B)_k)

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float8E4M3FN/Float8E5M2
          A and B can have different types (Float8E4M3FN/Float8E5M2)
          only support k-major layout

    :note: Supported accumulation types:
        - Float32

    :note: Constraints:
        - CTA tile M must be 128
        - CTA tile N must be 128/256
        - CTA tile K must be 128
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # DeepGEMM's Hopper 1d2d path uses two math warp-groups for BLOCK_M > 64 so
        # the CUDA-core scale promotion sees the full 128-row block instead of a
        # single 64-row wave. Mirror that here.
        self.atom_layout_mnk = (
            (2, 1, 1) if self.tile_shape_mnk[0] > 64 else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.sfa_smem_layout_staged = None
        self.sfb_smem_layout = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.scale_n_groups_per_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024
        self.scale_granularity_m = SCALE_GRANULARITY_M
        self.scale_granularity_n = SCALE_GRANULARITY_N
        self.scale_granularity_k = SCALE_GRANULARITY_K
        self.scale_dtype = SCALE_DTYPE

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        if self.tile_shape_mnk[0] != 128:
            raise ValueError("CTA tile shape M must be 128 for Hopper FP8 scaling")
        if self.tile_shape_mnk[1] not in [128, 256]:
            raise ValueError("CTA tile shape N must be 128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        if self.tile_shape_mnk[2] != self.scale_granularity_k:
            raise ValueError(
                "This kernel expects a Hopper FP8 K tile of 128 to match scale promotion"
            )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.scale_n_groups_per_tile = math.ceil(
            self.tile_shape_mnk[1] / self.scale_granularity_n
        )

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )
        self.sfa_smem_layout_staged = self._make_sfa_smem_layout(
            self.tile_shape_mnk[0], self.ab_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.sfa_dtype = sfa.element_type
        self.sfb_dtype = sfb.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.sfa_dtype != self.scale_dtype):
            raise TypeError(f"sfa must have dtype {self.scale_dtype}")
        if cutlass.const_expr(self.sfb_dtype != self.scale_dtype):
            raise TypeError(f"sfb must have dtype {self.scale_dtype}")

        if cutlass.const_expr(self.a_layout != utils.LayoutEnum.ROW_MAJOR):
            raise TypeError("A must be row-major / k-major")
        if cutlass.const_expr(self.b_layout != utils.LayoutEnum.ROW_MAJOR):
            raise TypeError("B must be row-major / k-major")

        # Hopper's SFA operand uses a broadcasted blockscale layout rather than
        # a plain row-major `(M, Kgroups, L)` tensor. Rewrap the compact storage
        # so TMA sees the same logical layout as CUTLASS.
        sfa = cute.make_tensor(
            sfa.iterator,
            self._make_sfa_gmem_layout(a.shape[0], a.shape[1], a.shape[2]),
        )

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa,
            self.sfa_smem_layout_staged,
            (self.tile_shape_mnk[0], 1),
            self.cluster_shape_mn[1],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mn)
        self.sfb_smem_layout = cute.make_layout(
            (self.scale_n_groups_per_tile, sfb.shape[1]),
            stride=(sfb.shape[1], 1),
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.scale_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                128,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.scale_dtype, cute.cosize(self.sfb_smem_layout)
                ],
                16,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            sfb,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        # CTA Swizzle to promote L2 data reuse
        group_size_m = 8
        s_shape = (
            (group_size_m, cdimx // group_size_m),
            cdimy,
        )
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

        # Deal with the tail part
        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]

        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        tma_copy_bytes += cute.size_in_bytes(self.scale_dtype, sfa_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        #  Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, RestK)
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        # (bN, bK, RestK)
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        # (bM, 1, RestK)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            (self.tile_shape_mnk[0], 1, 1),
            tile_coord_mnkl,
            proj=(1, None, 1),
        )
        # (bM, bN)
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        num_sfb = self.scale_n_groups_per_tile * k_tile_cnt

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mnl)
        cC = cute.make_identity_tensor((self.tile_shape_mnk[0], self.tile_shape_mnk[1]))
        tCcC = thr_mma.partition_C(cC)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma_partition,
            gB_for_tma_partition,
        )

        # TMA load SFA partition_S/D
        tAsSFA, tAgSFA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA_mkl = cute.filter_zeros(tAgSFA_mkl)

        # //////////////////////////////////////////////////////////////////////////////
        #  Make fragments
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_shape = thr_mma.partition_shape_C(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        )
        # Both accumulator tensors live in RMEM. `accumulators_tmp` is the
        # per-128-K-group WGMMA result, while `accumulators` is the promoted
        # FP32 output accumulator.
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        accumulators.fill(0.0)
        accumulators_tmp = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        copy_atom_r2s_acc = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C_acc = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )
        tiled_copy_C_acc = cute.make_tiled_copy_C_atom(copy_atom_C_acc, tiled_mma)
        tiled_copy_r2s_acc = cute.make_tiled_copy_S(
            copy_atom_r2s_acc,
            tiled_copy_C_acc,
        )
        thr_copy_r2s_acc = tiled_copy_r2s_acc.get_slice(tidx)
        # Retile only creates register views that match the store path's logical
        # (M, N) ownership. No extra SMEM or GMEM copies happen here.
        tPromote_rAcc = tiled_copy_r2s_acc.retile(accumulators)
        tPromote_rAcc_tmp = tiled_copy_r2s_acc.retile(accumulators_tmp)
        tPromote_cC = thr_copy_r2s_acc.partition_S(cC)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cluster wait
        # ///////////////////////////////////////////////////////////////////////////////
        # cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # /////////////////////////////////////////////////////////////////////////////
        #  Preload the compact B scales for this CTA tile
        # /////////////////////////////////////////////////////////////////////////////
        for sfb_linear_idx in cutlass.range(tidx, num_sfb, self.threads_per_cta, unroll=1):
            local_scale_n = sfb_linear_idx // k_tile_cnt
            scale_k = sfb_linear_idx - local_scale_n * k_tile_cnt
            global_scale_n = pid_n * self.scale_n_groups_per_tile + local_scale_n
            if cute.elem_less((global_scale_n, scale_k, bidz), mSFB_nkl.shape):
                sSFB[(local_scale_n, scale_k)] = mSFB_nkl[
                    (global_scale_n, scale_k, bidz)
                ]
            else:
                sSFB[(local_scale_n, scale_k)] = 0.0
        cute.arch.sync_threads()
        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        # /////////////////////////////////////////////////////////////////////////////
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if warp_idx == 0:
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                tAgSFA_k = tAgSFA_mkl[(None, mainloop_producer_state.count)]
                tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B/SFA
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA_k,
                    tAsSFA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        # Hopper FP8 scaling promotes every 128-wide K group, so we keep a
        # single temporary RMEM fragment for the current WGMMA group while still
        # tracking the shared-memory stage with separate read/release cursors.
        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        num_k_blocks = cute.size(tCrA, mode=[2])
        tile_m_offset = pid_m * self.tile_shape_mnk[0]
        tile_n_offset = pid_n * self.tile_shape_mnk[1]
        problem_shape_mn = (mC_mnl.shape[0], mC_mnl.shape[1])

        for _ in cutlass.range(k_tile_cnt, unroll=1):
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            # /////////////////////////////////////////////////////////////////////////////
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA for the current 128-wide K group
            # /////////////////////////////////////////////////////////////////////////////
            accumulators_tmp.fill(0.0)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators_tmp,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators_tmp,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            # /////////////////////////////////////////////////////////////////////////////
            #  Promote the temporary RMEM fragment with staged scales
            # /////////////////////////////////////////////////////////////////////////////
            for i in cutlass.range_constexpr(cute.size(tPromote_rAcc.shape)):
                local_coord = tPromote_cC[i]
                global_row = tile_m_offset + local_coord[0]
                global_col = tile_n_offset + local_coord[1]
                if cute.elem_less((global_row, global_col), problem_shape_mn):
                    scale_a = sSFA[
                        (local_coord[0], 0, mainloop_consumer_release_state.index)
                    ].to(self.acc_dtype)
                    scale_b = sSFB[
                        (
                            local_coord[1] // self.scale_granularity_n,
                            mainloop_consumer_release_state.count,
                        )
                    ].to(self.acc_dtype)
                    tPromote_rAcc[i] = (
                        tPromote_rAcc[i] + tPromote_rAcc_tmp[i] * (scale_a * scale_b)
                    )

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            # /////////////////////////////////////////////////////////////////////////////
            #  Peek on the next full stage
            # /////////////////////////////////////////////////////////////////////////////
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load A/B/SFA
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B/SFA buffers to be empty before loading into them
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                tAgSFA_k = tAgSFA_mkl[(None, mainloop_producer_state.count)]
                tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA_k,
                    tAsSFA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        # /////////////////////////////////////////////////////////////////////////////
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mn) > 1:
            # Wait for all threads in the cluster to finish, avoid early release of smem
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            # For cluster that has a single thread block, it might have more than one warp groups.
            # Wait for all warp groups in the thread block to finish, because smem for tensor A in
            # the mainloop is reused in the epilogue.
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        # Initialize tma store c_pipeline
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from accumulators to D registers
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            # Type conversion
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # barrier for sync
            pipeline.sync(barrier_id=1)

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()

            pipeline.sync(barrier_id=1)

        if warp_idx == 0:
            c_pipeline.producer_tail()

        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        epi_stage = 4
        # epi_smem will reuse smem ab.
        epi_bytes = 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        # SFA is staged alongside A/B in the Hopper mainloop.
        ab_bytes_per_stage += tile_shape_mnk[0] * SCALE_DTYPE.width // 8
        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_sfa_smem_layout(tile_m: int, ab_stage: int) -> cute.Layout:
        return cute.make_layout(
            (tile_m, 1, ab_stage),
            stride=(1, tile_m, tile_m),
        )

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl)
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))
        return grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.Layout | cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def _make_sfa_gmem_layout(m: int, k: int, l: int) -> cute.Layout:
        """Create the CUTLASS logical gmem layout for Hopper SFA tensors."""
        m_groups = cute.ceil_div(m, SCALE_GRANULARITY_M)
        k_groups = cute.ceil_div(k, SCALE_GRANULARITY_K)
        mk_layout = cute.make_layout(
            (
                (SCALE_GRANULARITY_M, m_groups),
                (SCALE_GRANULARITY_K, k_groups),
            ),
            stride=((0, 1), (0, m_groups)),
        )
        return cute.make_layout(
            cute.append(mk_layout.shape, l),
            stride=cute.append(
                mk_layout.stride, cute.size(cute.filter_zeros(mk_layout))
            ),
        )

    @staticmethod
    def expected_scale_shapes(
        m: int, n: int, k: int, l: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        k_groups = math.ceil(k / SCALE_GRANULARITY_K)
        n_groups = math.ceil(n / SCALE_GRANULARITY_N)
        return (m, k_groups, l), (n_groups, k_groups, l)

    @staticmethod
    def layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())

        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])

        if cutlass.const_expr(cute.rank(lt) == 1):
            return cute.append(lt, ge)
        return cute.append(cute.append(cute.make_layout(()), lt), ge)

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
        )

        v_m = separated[0]
        v_n = separated[1]
        if cutlass.const_expr(cute.rank(v_m) == 1):
            v_m1 = cute.append(v_m, acc[1])
        else:
            v_m1 = cute.append(cute.append(cute.make_layout(()), v_m), acc[1])

        if cutlass.const_expr(cute.rank(v_n) == 1):
            v_n1 = cute.append(v_n, acc[2])
        else:
            v_n1 = cute.append(cute.append(cute.make_layout(()), v_n), acc[2])

        if cutlass.const_expr(cute.rank(v_m1) == 1):
            return cute.append(v_m1, v_n1)
        return cute.append(cute.append(cute.make_layout(()), v_m1), v_n1)

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        valid_ab_dtypes = {
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }
        valid_c_dtypes = {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }

        return (
            a_dtype in valid_ab_dtypes
            and b_dtype in valid_ab_dtypes
            and acc_dtype == cutlass.Float32
            and c_dtype in valid_c_dtypes
            and a_major == "k"
            and b_major == "k"
        )

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid


def create_tensors(
    l: int,
    m: int,
    n: int,
    k: int,
    a_major: str,
    b_major: str,
    c_major: str,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    scale_dtype: Type[cutlass.Numeric],
):
    import torch
    import cutlass.torch as cutlass_torch

    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", a_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", b_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major == "m", c_dtype)
    sfa_torch_cpu = torch.empty(
        (m, math.ceil(k / SCALE_GRANULARITY_K), l), dtype=cutlass_torch.dtype(scale_dtype)
    ).uniform_(-1.0, 1.0)
    sfb_torch_cpu = torch.empty(
        (math.ceil(n / SCALE_GRANULARITY_N), math.ceil(k / SCALE_GRANULARITY_K), l),
        dtype=cutlass_torch.dtype(scale_dtype),
    ).uniform_(-1.0, 1.0)

    a_tensor, a_torch_gpu = cutlass_torch.cute_tensor_like(
        a_torch_cpu, a_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch_gpu = cutlass_torch.cute_tensor_like(
        b_torch_cpu, b_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )
    sfa_torch_gpu = pack_sfa_tensor_for_hopper(sfa_torch_cpu.cuda())
    sfb_torch_gpu = sfb_torch_cpu.cuda()
    sfa_tensor = from_dlpack(sfa_torch_gpu, assumed_align=16)
    sfb_tensor = from_dlpack(sfb_torch_gpu, assumed_align=16)

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_gpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
    )


def expand_scale(scale, tensor, mode0_granularity: int):
    import torch

    mode0_extent, k_extent, _ = tensor.shape
    mode0_idx = torch.arange(mode0_extent, device=scale.device) // mode0_granularity
    k_idx = torch.arange(k_extent, device=scale.device) // SCALE_GRANULARITY_K
    return scale[mode0_idx[:, None], k_idx, :] * tensor


def pack_sfa_tensor_for_hopper(sfa):
    """Pack a logical `(M, ceil_div(K, 128), L)` torch tensor into Hopper SFA storage."""
    import torch

    if sfa.ndim != 3:
        raise ValueError("sfa must have shape (M, ceil_div(K, 128), L)")
    m, k_groups, l = sfa.shape
    packed = torch.empty_strided(
        (m, k_groups, l),
        (1, m, m * k_groups),
        device=sfa.device,
        dtype=sfa.dtype,
    )
    packed.copy_(sfa)
    return packed


def run(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    import torch
    import cutlass.torch as cutlass_torch

    print("Running Hopper FP8 Groupwise GEMM with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Scale dtype: {SCALE_DTYPE}")
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")

    m, n, k, l = mnkl

    if not HopperWgmmaGemmKernel.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
    ):
        raise TypeError(
            f"unsupported combination of types and majors: A {a_dtype}, B {b_dtype}, Acc {acc_dtype}, C {c_dtype}, {a_major=}, {b_major=}"
        )
    if not HopperWgmmaGemmKernel.is_valid_tensor_alignment(
        m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
    ):
        raise TypeError(
            "the contiguous dimension of A/B/C tensors is not 16 bytes aligned"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    (
        mA,
        mB,
        mC,
        mSFA,
        mSFB,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_gpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
    ) = create_tensors(
        l, m, n, k, a_major, b_major, c_major, a_dtype, b_dtype, c_dtype, SCALE_DTYPE
    )

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    if k % SCALE_GRANULARITY_K != 0:
        raise ValueError("This implementation currently requires K to be a multiple of 128")

    dense_kernel = HopperWgmmaGemmKernel(
        cutlass.Float32, tile_shape_mn, cluster_shape_mn
    )
    compiled_dense_kernel = cute.compile(
        dense_kernel,
        mA,
        mB,
        mSFA,
        mSFB,
        mC,
        stream,
    )

    def launch_kernel():
        compiled_dense_kernel(
            mA,
            mB,
            mSFA,
            mSFB,
            mC,
            stream,
    )

    if not skip_ref_check:
        launch_kernel()
        torch.cuda.synchronize()

        updated_a = expand_scale(sfa_torch_cpu, a_torch_cpu, SCALE_GRANULARITY_M)
        updated_b = expand_scale(sfb_torch_cpu, b_torch_cpu, SCALE_GRANULARITY_N)
        ref = torch.einsum("mkl,nkl->mnl", updated_a, updated_b).to(
            cutlass_torch.dtype(c_dtype)
        )
        res = c_torch_gpu.view(cutlass_torch.dtype(c_dtype))
        torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-03)

    for _ in range(warmup_iterations):
        launch_kernel()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch_kernel()
    end.record()
    torch.cuda.synchronize()

    exec_time_ms = start.elapsed_time(end) / max(iterations, 1)
    return exec_time_ms * 1000.0


if __name__ == "__main__":
    args = parse_arguments()
    run(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mn,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
