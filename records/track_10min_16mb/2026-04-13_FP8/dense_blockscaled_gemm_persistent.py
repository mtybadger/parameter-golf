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
import math
from typing import Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

import dense_blockscaled_gemm as base

"""
A persistent Hopper FP8 grouped/blockscaled GEMM example in CuTe DSL.

This version keeps the same grouped/blockscaled math as
`dense_blockscaled_gemm.py`, but moves the kernel structure closer to the
Hopper persistent examples and DeepGEMM SM90 1d2d:
- one dedicated DMA warp-group,
- one or two math warp-groups depending on the CTA tile,
- persistent tile scheduling,
- TMA staging for A/B/SFA,
- staged preload of SFB per work tile by the math warp-groups.

The current implementation targets the same compact Hopper scale tensors as the
non-persistent kernel:
- SFA shape `(M, ceil_div(K, 128), L)` packed with stride `(1, M, M * Kgroups)`
- SFB shape `(ceil_div(N, 128), ceil_div(K, 128), L)`
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent Hopper FP8 grouped/blockscaled MxNxKxL GEMM."
    )

    parser.add_argument(
        "--mnkl",
        type=base.parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mn",
        type=base.parse_comma_separated_ints,
        choices=[(128, 128), (128, 256)],
        default=(128, 128),
        help="CTA tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=base.parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--swizzle_size",
        type=int,
        default=1,
        help="Persistent scheduler swizzle size in the unit of clusters",
    )
    parser.add_argument(
        "--raster_order",
        type=str,
        choices=["along_m", "along_n"],
        default="along_m",
        help="Persistent scheduler raster order",
    )
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
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
        "--iterations", type=int, default=1, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference validation"
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


class HopperWgmmaGemmPersistentKernel(base.HopperWgmmaGemmKernel):
    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
    ):
        super().__init__(acc_dtype, tile_shape_mn, cluster_shape_mn)

        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m

        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )

        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )
        self.scale_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_mma_threads
        )

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        epi_stage = 4
        epi_bytes = cute.size(epi_tile) * c_dtype.width // 8 * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        ab_bytes_per_stage += tile_shape_mnk[0] * base.SCALE_DTYPE.width // 8

        mbar_helpers_bytes = 1024
        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] != 128:
            raise ValueError("CTA tile shape M must be 128 for Hopper FP8 scaling")
        if self.tile_shape_mnk[1] not in [128, 256]:
            raise ValueError("CTA tile shape N must be 128/256")

        self.tiled_mma = base.sm90_utils.make_trivial_tiled_mma(
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
        self.epi_tile = base.sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        # The persistent blockscaled path adds staged SFA and a separate epilogue
        # buffer on top of the dense persistent scaffold. Cap the wider 128x256 CTA
        # to two mainloop stages so the launch stays within Hopper resource limits.
        if self.tile_shape_mnk[1] == 256:
            self.ab_stage = min(self.ab_stage, 2)

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

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
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

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
            max_active_clusters,
        )

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
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

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
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

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
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

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
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.scale_dtype, sfa_smem_layout)
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            (self.tile_shape_mnk[0], 1, 1),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        num_sfb = self.scale_n_groups_per_tile * k_tile_cnt

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            # `local_tile` on the logical Hopper SFA tensor preserves a singleton
            # tile-L mode. Group it with the vectorized tile modes so the producer
            # slices `tAgSFA` by (RestM, RestK, RestL) just like A/B.
            cute.group_modes(gSFA_mkl, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_tile = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_tile = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                tAgSFA_tile = tAgSFA[
                    (None, tile_coord_mnl[0], None, tile_coord_mnl[2])
                ]

                mainloop_producer_state.reset_count()
                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    cute.copy(
                        tma_atom_a,
                        tAgA_tile[(None, mainloop_producer_state.count)],
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_tile[(None, mainloop_producer_state.count)],
                        tBsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_tile[(None, mainloop_producer_state.count)],
                        tAsSFA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )

                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            mma_warp_group_thread_layout = cute.make_layout(
                self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(
                mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
            )

            cC = cute.make_identity_tensor((self.tile_shape_mnk[0], self.tile_shape_mnk[1]))
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCgC = thr_mma.partition_C(gC_mnl)

            acc_shape = thr_mma.partition_shape_C(
                (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
            )
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            accumulators_tmp = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            copy_atom_r2s = base.sm90_utils.sm90_get_smem_store_op(
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
            tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)

            math_thread_idx = tidx - (
                self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            thr_copy_r2s = tiled_copy_r2s.get_slice(math_thread_idx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            tRS_rAcc_tmp = tiled_copy_r2s.retile(accumulators_tmp)
            tPromote_cC = thr_copy_r2s.partition_S(cC)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_threads
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])
            lane_idx = cute.arch.lane_idx()
            math_warp_idx = (
                warp_idx - self.num_dma_warp_groups * self.num_warps_per_warp_group
            )
            scale_row_0 = math_warp_idx * 16 + lane_idx // 4
            scale_row_1 = scale_row_0 + 8

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]

                tile_m_offset = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                tile_n_offset = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                problem_shape_mn = (mC_mnl.shape[0], mC_mnl.shape[1])
                tile_is_full = (
                    tile_m_offset + self.tile_shape_mnk[0] <= problem_shape_mn[0]
                    and tile_n_offset + self.tile_shape_mnk[1] <= problem_shape_mn[1]
                )

                for sfb_linear_idx in cutlass.range(
                    math_thread_idx, num_sfb, self.num_mma_threads, unroll=1
                ):
                    local_scale_n = sfb_linear_idx // k_tile_cnt
                    scale_k = sfb_linear_idx - local_scale_n * k_tile_cnt
                    global_scale_n = (
                        tile_coord_mnl[1] * self.scale_n_groups_per_tile + local_scale_n
                    )
                    if cute.elem_less(
                        (global_scale_n, scale_k, tile_coord_mnl[2]), mSFB_nkl.shape
                    ):
                        sSFB[(local_scale_n, scale_k)] = mSFB_nkl[
                            (global_scale_n, scale_k, tile_coord_mnl[2])
                        ]
                    else:
                        sSFB[(local_scale_n, scale_k)] = 0.0
                self.scale_sync_barrier.arrive_and_wait()

                accumulators.fill(0.0)
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()

                for _ in range(k_tile_cnt):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)

                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators_tmp,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators_tmp,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    scale_stage_idx = mainloop_consumer_release_state.index
                    scale_k_idx = mainloop_consumer_release_state.count
                    scale_a_0 = self.acc_dtype(0.0)
                    scale_a_1 = self.acc_dtype(0.0)
                    if tile_is_full:
                        scale_a_0 = sSFA[(scale_row_0, 0, scale_stage_idx)].to(
                            self.acc_dtype
                        )
                        scale_a_1 = sSFA[(scale_row_1, 0, scale_stage_idx)].to(
                            self.acc_dtype
                        )

                    if tile_is_full and self.scale_n_groups_per_tile == 1:
                        scale_b = sSFB[(0, scale_k_idx)].to(self.acc_dtype)
                        scale_0 = scale_a_0 * scale_b
                        scale_1 = scale_a_1 * scale_b
                        for i in cutlass.range_constexpr(
                            0, cute.size(tRS_rAcc.shape), 4
                        ):
                            tRS_rAcc[i + 0] = (
                                tRS_rAcc[i + 0] + tRS_rAcc_tmp[i + 0] * scale_0
                            )
                            tRS_rAcc[i + 1] = (
                                tRS_rAcc[i + 1] + tRS_rAcc_tmp[i + 1] * scale_0
                            )
                            tRS_rAcc[i + 2] = (
                                tRS_rAcc[i + 2] + tRS_rAcc_tmp[i + 2] * scale_1
                            )
                            tRS_rAcc[i + 3] = (
                                tRS_rAcc[i + 3] + tRS_rAcc_tmp[i + 3] * scale_1
                            )
                    else:
                        for i in cutlass.range_constexpr(
                            0, cute.size(tRS_rAcc.shape), 4
                        ):
                            coord_00 = tPromote_cC[i + 0]
                            coord_01 = tPromote_cC[i + 1]
                            coord_10 = tPromote_cC[i + 2]
                            coord_11 = tPromote_cC[i + 3]

                            if not tile_is_full:
                                scale_a_0 = sSFA[(coord_00[0], 0, scale_stage_idx)].to(
                                    self.acc_dtype
                                )
                                scale_a_1 = sSFA[(coord_10[0], 0, scale_stage_idx)].to(
                                    self.acc_dtype
                                )
                            scale_b_0 = sSFB[
                                (coord_00[1] // self.scale_granularity_n, scale_k_idx)
                            ].to(self.acc_dtype)
                            scale_b_1 = sSFB[
                                (coord_01[1] // self.scale_granularity_n, scale_k_idx)
                            ].to(self.acc_dtype)

                            scale_00 = scale_a_0 * scale_b_0
                            scale_01 = scale_a_0 * scale_b_1
                            scale_10 = scale_a_1 * scale_b_0
                            scale_11 = scale_a_1 * scale_b_1

                            if tile_is_full:
                                tRS_rAcc[i + 0] = (
                                    tRS_rAcc[i + 0]
                                    + tRS_rAcc_tmp[i + 0] * scale_00
                                )
                                tRS_rAcc[i + 1] = (
                                    tRS_rAcc[i + 1]
                                    + tRS_rAcc_tmp[i + 1] * scale_01
                                )
                                tRS_rAcc[i + 2] = (
                                    tRS_rAcc[i + 2]
                                    + tRS_rAcc_tmp[i + 2] * scale_10
                                )
                                tRS_rAcc[i + 3] = (
                                    tRS_rAcc[i + 3]
                                    + tRS_rAcc_tmp[i + 3] * scale_11
                                )
                            else:
                                global_coord_00 = (
                                    tile_m_offset + coord_00[0],
                                    tile_n_offset + coord_00[1],
                                )
                                global_coord_01 = (
                                    tile_m_offset + coord_01[0],
                                    tile_n_offset + coord_01[1],
                                )
                                global_coord_10 = (
                                    tile_m_offset + coord_10[0],
                                    tile_n_offset + coord_10[1],
                                )
                                global_coord_11 = (
                                    tile_m_offset + coord_11[0],
                                    tile_n_offset + coord_11[1],
                                )

                                if cute.elem_less(global_coord_00, problem_shape_mn):
                                    tRS_rAcc[i + 0] = (
                                        tRS_rAcc[i + 0]
                                        + tRS_rAcc_tmp[i + 0] * scale_00
                                    )
                                if cute.elem_less(global_coord_01, problem_shape_mn):
                                    tRS_rAcc[i + 1] = (
                                        tRS_rAcc[i + 1]
                                        + tRS_rAcc_tmp[i + 1] * scale_01
                                    )
                                if cute.elem_less(global_coord_10, problem_shape_mn):
                                    tRS_rAcc[i + 2] = (
                                        tRS_rAcc[i + 2]
                                        + tRS_rAcc_tmp[i + 2] * scale_10
                                    )
                                if cute.elem_less(global_coord_11, problem_shape_mn):
                                    tRS_rAcc[i + 3] = (
                                        tRS_rAcc[i + 3]
                                        + tRS_rAcc_tmp[i + 3] * scale_11
                                    )

                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                tCgC_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma_partition,
                )

                epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )
                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                        tRS_sD, mode=[3]
                    )
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                    self.epilog_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()


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
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    **kwargs,
):
    import torch
    import cutlass.torch as cutlass_torch

    print("Running Hopper Persistent FP8 Groupwise GEMM with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Scale dtype: {base.SCALE_DTYPE}")
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(
        f"Swizzle size: {swizzle_size}, Raster order:",
        "along_m" if raster_along_m else "along_n",
    )
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")

    m, n, k, l = mnkl

    if not HopperWgmmaGemmPersistentKernel.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, c_dtype, a_major, b_major
    ):
        raise TypeError(
            f"unsupported combination of types and majors: A {a_dtype}, B {b_dtype}, Acc {acc_dtype}, C {c_dtype}, {a_major=}, {b_major=}"
        )
    if not HopperWgmmaGemmPersistentKernel.is_valid_tensor_alignment(
        m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
    ):
        raise TypeError(
            "the contiguous dimension of A/B/C tensors is not 16 bytes aligned"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    if k % base.SCALE_GRANULARITY_K != 0:
        raise ValueError("This implementation currently requires K to be a multiple of 128")

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
        _a_torch_gpu,
        _b_torch_gpu,
        _sfa_torch_gpu,
        _sfb_torch_gpu,
    ) = base.create_tensors(
        l, m, n, k, a_major, b_major, c_major, a_dtype, b_dtype, c_dtype, base.SCALE_DTYPE
    )

    gemm = HopperWgmmaGemmPersistentKernel(
        acc_dtype, tile_shape_mn, cluster_shape_mn, swizzle_size, raster_along_m
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled_gemm = cute.compile(
        gemm,
        mA,
        mB,
        mSFA,
        mSFB,
        mC,
        max_active_clusters,
        stream,
    )

    def launch_kernel():
        compiled_gemm(mA, mB, mSFA, mSFB, mC, stream)

    if not skip_ref_check:
        launch_kernel()
        torch.cuda.synchronize()

        updated_a = base.expand_scale(
            sfa_torch_cpu, a_torch_cpu, base.SCALE_GRANULARITY_M
        )
        updated_b = base.expand_scale(
            sfb_torch_cpu, b_torch_cpu, base.SCALE_GRANULARITY_N
        )
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
        args.swizzle_size,
        args.raster_order == "along_m",
    )
    print("PASS")
