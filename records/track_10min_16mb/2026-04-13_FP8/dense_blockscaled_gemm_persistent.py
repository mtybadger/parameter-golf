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
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.cpp_extension import load as torch_cpp_load
from cutlass.cute.runtime import from_dlpack
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
        self.sfb_k_groups = None

    @staticmethod
    def _compute_stage_upper_bound(
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

    @staticmethod
    def _align_up(size: int, alignment: int) -> int:
        return ((size + alignment - 1) // alignment) * alignment

    @staticmethod
    def _layout_size_in_bytes(
        layout: cute.Layout | cute.ComposedLayout,
        dtype: type[cutlass.Numeric],
    ) -> int:
        return cute.cosize(layout) * dtype.width // 8

    @staticmethod
    def _make_sfb_smem_layout(
        scale_n_groups_per_tile: int,
        sfb_k_groups: int,
    ) -> cute.Layout:
        return cute.make_layout(
            (scale_n_groups_per_tile, sfb_k_groups),
            stride=(sfb_k_groups, 1),
        )

    def _compute_shared_storage_bytes(
        self,
        ab_stage: int,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ) -> int:
        total = 0
        total = self._align_up(total, 8)
        total += ab_stage * 2 * 8

        total = self._align_up(total, self.buffer_align_bytes)
        total += self._layout_size_in_bytes(a_smem_layout_staged, self.a_dtype)

        total = self._align_up(total, self.buffer_align_bytes)
        total += self._layout_size_in_bytes(b_smem_layout_staged, self.b_dtype)

        total = self._align_up(total, 128)
        total += self._layout_size_in_bytes(sfa_smem_layout_staged, self.scale_dtype)

        total = self._align_up(total, 16)
        total += self._layout_size_in_bytes(sfb_smem_layout, self.scale_dtype)

        total = self._align_up(total, self.buffer_align_bytes)
        total += self._layout_size_in_bytes(epi_smem_layout_staged, self.c_dtype)
        return total

    def _select_stages_and_layouts(
        self,
    ) -> tuple[
        int,
        int,
        cute.ComposedLayout,
        cute.ComposedLayout,
        cute.Layout,
        cute.Layout,
        cute.ComposedLayout,
    ]:
        if self.sfb_k_groups is None:
            raise RuntimeError("sfb_k_groups must be set before stage selection")

        upper_bound, epi_stage = self._compute_stage_upper_bound(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        upper_bound = max(1, upper_bound)

        sfb_smem_layout = self._make_sfb_smem_layout(
            self.scale_n_groups_per_tile, self.sfb_k_groups
        )

        for ab_stage in range(upper_bound, 0, -1):
            (
                a_smem_layout_staged,
                b_smem_layout_staged,
                epi_smem_layout_staged,
            ) = self._make_smem_layouts(
                self.tile_shape_mnk,
                self.epi_tile,
                self.a_dtype,
                self.a_layout,
                self.b_dtype,
                self.b_layout,
                ab_stage,
                self.c_dtype,
                self.c_layout,
                epi_stage,
            )
            sfa_smem_layout_staged = self._make_sfa_smem_layout(
                self.tile_shape_mnk[0], ab_stage
            )

            total_bytes = self._compute_shared_storage_bytes(
                ab_stage,
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout,
                epi_smem_layout_staged,
            )
            if total_bytes <= self.smem_capacity:
                return (
                    ab_stage,
                    epi_stage,
                    a_smem_layout_staged,
                    b_smem_layout_staged,
                    sfa_smem_layout_staged,
                    sfb_smem_layout,
                    epi_smem_layout_staged,
                )

        raise RuntimeError("Failed to fit blockscaled persistent shared storage on SM90")

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

        (
            self.ab_stage,
            self.epi_stage,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout,
            self.epi_smem_layout_staged,
        ) = self._select_stages_and_layouts()
        # Use exact shared-memory accounting for the blockscaled persistent path
        # instead of a dense-kernel heuristic plus tile-specific caps.

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
        self.sfb_k_groups = sfb.shape[1]

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

                    if cutlass.const_expr(self.scale_n_groups_per_tile == 1):
                        if tile_is_full:
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

                                scale_a_0 = sSFA[(coord_00[0], 0, scale_stage_idx)].to(
                                    self.acc_dtype
                                )
                                scale_a_1 = sSFA[(coord_10[0], 0, scale_stage_idx)].to(
                                    self.acc_dtype
                                )
                                scale_b_0 = sSFB[(0, scale_k_idx)].to(self.acc_dtype)
                                scale_b_1 = sSFB[(0, scale_k_idx)].to(self.acc_dtype)

                                scale_00 = scale_a_0 * scale_b_0
                                scale_01 = scale_a_0 * scale_b_1
                                scale_10 = scale_a_1 * scale_b_0
                                scale_11 = scale_a_1 * scale_b_1

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
                    elif cutlass.const_expr(self.scale_n_groups_per_tile == 2):
                        scale_b_0 = sSFB[(0, scale_k_idx)].to(self.acc_dtype)
                        scale_b_1 = sSFB[(1, scale_k_idx)].to(self.acc_dtype)
                        if tile_is_full:
                            # For the Hopper 128x256 tile, B scales split at a
                            # fixed 128-column boundary.
                            former_scale_quads = self.scale_granularity_n // 8
                            for i in cutlass.range_constexpr(
                                0, cute.size(tRS_rAcc.shape), 4
                            ):
                                quad_idx = i // 4
                                quad_scale_b = (
                                    scale_b_0
                                    if quad_idx < former_scale_quads
                                    else scale_b_1
                                )
                                scale_0 = scale_a_0 * quad_scale_b
                                scale_1 = scale_a_1 * quad_scale_b
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

                                scale_a_0 = sSFA[(coord_00[0], 0, scale_stage_idx)].to(
                                    self.acc_dtype
                                )
                                scale_a_1 = sSFA[(coord_10[0], 0, scale_stage_idx)].to(
                                    self.acc_dtype
                                )
                                quad_scale_b_0 = (
                                    scale_b_0
                                    if coord_00[1] < self.scale_granularity_n
                                    else scale_b_1
                                )
                                quad_scale_b_1 = (
                                    scale_b_0
                                    if coord_01[1] < self.scale_granularity_n
                                    else scale_b_1
                                )

                                scale_00 = scale_a_0 * quad_scale_b_0
                                scale_01 = scale_a_0 * quad_scale_b_1
                                scale_10 = scale_a_1 * quad_scale_b_0
                                scale_11 = scale_a_1 * quad_scale_b_1

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


FP8_E4M3_MAX = 448.0
DEFAULT_TRAIN_GPT_PATH = Path(__file__).with_name("train_gpt.py")
_TRAIN_GPT_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "MODEL_DIM",
    "MLP_MULT",
)


@dataclass
class BlockScaledQuantizedOperand:
    data: Tensor
    logical_scale: Tensor
    kernel_scale: Tensor


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    mapping = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported output dtype for persistent blockscaled GEMM: {dtype}")
    return mapping[dtype]


def _cuda_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _scale_from_amax(amax: Tensor) -> Tensor:
    scale = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    return scale.clamp_min(1e-12)


def quantize_blockscaled_a(x_2d: Tensor) -> BlockScaledQuantizedOperand:
    if x_2d.ndim != 2:
        raise ValueError(f"A operand must be 2D, got shape {tuple(x_2d.shape)}")
    m, k = x_2d.shape
    k_groups = _ceil_div(k, base.SCALE_GRANULARITY_K)
    k_padded = k_groups * base.SCALE_GRANULARITY_K
    x_padded = F.pad(x_2d.float(), (0, k_padded - k))
    x_blocks = x_padded.view(m, k_groups, base.SCALE_GRANULARITY_K)
    logical_scale = _scale_from_amax(x_blocks.abs().amax(dim=-1, keepdim=True))
    q_blocks = (x_blocks / logical_scale).to(torch.float8_e4m3fn)
    q = q_blocks.reshape(m, k_padded)[:, :k].contiguous().unsqueeze(-1)
    logical_scale = logical_scale.to(torch.float32)
    kernel_scale = base.pack_sfa_tensor_for_hopper(logical_scale)
    return BlockScaledQuantizedOperand(q, logical_scale, kernel_scale)


def quantize_blockscaled_b(x_2d: Tensor) -> BlockScaledQuantizedOperand:
    if x_2d.ndim != 2:
        raise ValueError(f"B operand must be 2D, got shape {tuple(x_2d.shape)}")
    n, k = x_2d.shape
    n_groups = _ceil_div(n, base.SCALE_GRANULARITY_N)
    k_groups = _ceil_div(k, base.SCALE_GRANULARITY_K)
    n_padded = n_groups * base.SCALE_GRANULARITY_N
    k_padded = k_groups * base.SCALE_GRANULARITY_K
    x_padded = F.pad(x_2d.float(), (0, k_padded - k, 0, n_padded - n))
    x_blocks = x_padded.view(
        n_groups,
        base.SCALE_GRANULARITY_N,
        k_groups,
        base.SCALE_GRANULARITY_K,
    ).permute(0, 2, 1, 3)
    logical_scale = _scale_from_amax(
        x_blocks.abs().amax(dim=(2, 3), keepdim=True)
    ).to(torch.float32)
    q_blocks = (x_blocks / logical_scale).to(torch.float8_e4m3fn)
    q = q_blocks.permute(0, 2, 1, 3).reshape(n_padded, k_padded)[:, :k]
    logical_scale = logical_scale.squeeze(-1).squeeze(-1).unsqueeze(-1).contiguous()
    return BlockScaledQuantizedOperand(
        q.contiguous().unsqueeze(-1), logical_scale, logical_scale
    )


class PersistentBlockScaledGemmRunner:
    def __init__(
        self,
        tile_shape_mn: tuple[int, int] = (128, 256),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        swizzle_size: int = 1,
        raster_along_m: bool = True,
        a_dtype: type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        b_dtype: type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.tile_shape_mn = tile_shape_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.acc_dtype = acc_dtype
        hardware_info = cutlass.utils.HardwareInfo()
        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        self.max_active_clusters = hardware_info.get_max_active_clusters(cluster_size)
        self._compiled: dict[tuple, object] = {}

    @staticmethod
    def _signature(tensor: Tensor) -> tuple:
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)

    def quantize_a(self, x_2d: Tensor) -> BlockScaledQuantizedOperand:
        return quantize_blockscaled_a(x_2d)

    def quantize_b(self, x_2d: Tensor) -> BlockScaledQuantizedOperand:
        return quantize_blockscaled_b(x_2d)

    def _compile(
        self,
        a: BlockScaledQuantizedOperand,
        b: BlockScaledQuantizedOperand,
        c: Tensor,
    ):
        c_dtype = _torch_to_cutlass_dtype(c.dtype)
        m, k, l = a.data.shape
        n = b.data.shape[0]
        if not HopperWgmmaGemmPersistentKernel.is_valid_dtypes(
            self.a_dtype, self.b_dtype, self.acc_dtype, c_dtype, "k", "k"
        ):
            raise TypeError("Unsupported dtype combination for persistent blockscaled GEMM")
        if not HopperWgmmaGemmPersistentKernel.is_valid_tensor_alignment(
            m, n, k, l, self.a_dtype, c_dtype, "k", "k", "n"
        ):
            raise TypeError(
                f"Tensor alignment mismatch for problem {(m, n, k, l)} and output dtype {c.dtype}"
            )
        key = (
            self._signature(a.data),
            self._signature(b.data),
            self._signature(a.kernel_scale),
            self._signature(b.kernel_scale),
            self._signature(c),
            self.tile_shape_mn,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
        )
        compiled = self._compiled.get(key)
        if compiled is None:
            kernel = HopperWgmmaGemmPersistentKernel(
                self.acc_dtype,
                self.tile_shape_mn,
                self.cluster_shape_mn,
                self.swizzle_size,
                self.raster_along_m,
            )
            stream = _cuda_stream()
            compiled = cute.compile(
                kernel,
                from_dlpack(a.data, assumed_align=16),
                from_dlpack(b.data, assumed_align=16),
                from_dlpack(a.kernel_scale, assumed_align=16),
                from_dlpack(b.kernel_scale, assumed_align=16),
                from_dlpack(c, assumed_align=16),
                self.max_active_clusters,
                stream,
            )
            self._compiled[key] = compiled
        return compiled

    def matmul_quantized(
        self,
        a: BlockScaledQuantizedOperand,
        b: BlockScaledQuantizedOperand,
        out_dtype: torch.dtype,
    ) -> Tensor:
        if a.data.device.type != "cuda" or b.data.device.type != "cuda":
            raise RuntimeError("Persistent blockscaled GEMM expects CUDA tensors")
        if a.data.ndim != 3 or b.data.ndim != 3:
            raise ValueError("Quantized operands must be rank-3 tensors with trailing L dimension")
        if a.data.shape[1] != b.data.shape[1]:
            raise ValueError(
                f"Reduction dimension mismatch: {a.data.shape} vs {b.data.shape}"
            )
        if a.data.shape[2] != 1 or b.data.shape[2] != 1:
            raise ValueError("This linear wrapper expects L == 1")
        c = torch.empty(
            (a.data.shape[0], b.data.shape[0], 1),
            device=a.data.device,
            dtype=out_dtype,
        )
        compiled = self._compile(a, b, c)
        stream = _cuda_stream()
        compiled(
            from_dlpack(a.data, assumed_align=16),
            from_dlpack(b.data, assumed_align=16),
            from_dlpack(a.kernel_scale, assumed_align=16),
            from_dlpack(b.kernel_scale, assumed_align=16),
            from_dlpack(c, assumed_align=16),
            stream,
        )
        return c[..., 0]

    def matmul(
        self,
        a_2d: Tensor,
        b_2d: Tensor,
        out_dtype: torch.dtype | None = None,
        quantized_b: BlockScaledQuantizedOperand | None = None,
    ) -> tuple[Tensor, BlockScaledQuantizedOperand]:
        qa = self.quantize_a(a_2d.contiguous())
        qb = quantized_b if quantized_b is not None else self.quantize_b(b_2d.contiguous())
        out = self.matmul_quantized(qa, qb, out_dtype or a_2d.dtype)
        return out, qb


class PersistentBlockScaledLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        runner: PersistentBlockScaledGemmRunner,
    ) -> Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d, q_weight = runner.matmul(
            x_2d,
            weight.contiguous(),
            out_dtype=x.dtype,
        )
        if bias is not None:
            y_2d = y_2d + bias.to(y_2d.dtype)
        ctx.runner = runner
        ctx.input_shape = tuple(x.shape)
        ctx.output_shape = tuple(y_2d.shape)
        ctx.input_dtype = x.dtype
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.weight_dtype = weight.dtype
        ctx.save_for_backward(x_2d, weight.contiguous())
        return y_2d.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, weight = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_input_2d, _ = ctx.runner.matmul(
            grad_out_2d,
            weight.transpose(0, 1).contiguous(),
            out_dtype=ctx.input_dtype,
        )
        grad_weight, _ = ctx.runner.matmul(
            grad_out_2d.transpose(0, 1).contiguous(),
            x_2d.transpose(0, 1).contiguous(),
            out_dtype=ctx.weight_dtype,
        )
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_out_2d.sum(dim=0).to(ctx.bias_dtype)
        return grad_input_2d.view(ctx.input_shape), grad_weight, grad_bias, None


class PersistentBlockScaledLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        tile_shape_mn: tuple[int, int] = (128, 256),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        self.runner = PersistentBlockScaledGemmRunner(
            tile_shape_mn=tile_shape_mn,
            cluster_shape_mn=cluster_shape_mn,
            swizzle_size=swizzle_size,
            raster_along_m=raster_along_m,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        **runner_kwargs,
    ) -> "PersistentBlockScaledLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **runner_kwargs,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def forward(self, x: Tensor) -> Tensor:
        return PersistentBlockScaledLinearFunction.apply(
            x, self.weight, self.bias, self.runner
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


_CUTLASS_GROUPWISE_OP_NAME = "cutlass_groupwise_gemm_sm90a"
_CUTLASS_GROUPWISE_AUTOTUNE_CANDIDATES = (
    (0, 1),
    (1, 1),
    (2, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (0, 4),
    (1, 4),
    (2, 4),
)


def _cutlass_groupwise_source_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    return (
        base_dir / "cutlass_groupwise_gemm_op.cpp",
        base_dir / "cutlass_groupwise_gemm_op.cu",
    )


@lru_cache(maxsize=1)
def _load_cutlass_groupwise_custom_op():
    sources = _cutlass_groupwise_source_paths()
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = Path(__file__).resolve().parent / ".torch_extensions" / _CUTLASS_GROUPWISE_OP_NAME
    build_dir.mkdir(parents=True, exist_ok=True)
    venv_bin = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if venv_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join((venv_bin, *path_parts))
    return torch_cpp_load(
        name=_CUTLASS_GROUPWISE_OP_NAME,
        sources=[str(path) for path in sources],
        extra_include_paths=[
            str(repo_root / "cutlass" / "include"),
            str(repo_root / "cutlass" / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-DNDEBUG",
            "--use_fast_math",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )


def _quantize_cutlass_groupwise_a_tensors(x_2d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if x_2d.ndim != 2:
        raise ValueError(f"A operand must be 2D, got shape {tuple(x_2d.shape)}")
    m, k = x_2d.shape
    k_groups = _ceil_div(k, base.SCALE_GRANULARITY_K)
    k_padded = k_groups * base.SCALE_GRANULARITY_K
    x_padded = F.pad(x_2d.float(), (0, k_padded - k))
    x_blocks = x_padded.view(m, k_groups, base.SCALE_GRANULARITY_K)
    logical_scale = _scale_from_amax(x_blocks.abs().amax(dim=-1, keepdim=True)).to(
        torch.float32
    )
    q_blocks = (x_blocks / logical_scale).to(torch.float8_e4m3fn)
    q = q_blocks.reshape(m, k_padded)[:, :k].contiguous()
    kernel_scale = base.pack_sfa_tensor_for_hopper(logical_scale)
    return q, logical_scale, kernel_scale


def _quantize_cutlass_groupwise_b_tensors(x_2d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if x_2d.ndim != 2:
        raise ValueError(f"B operand must be 2D, got shape {tuple(x_2d.shape)}")
    n, k = x_2d.shape
    n_groups = _ceil_div(n, base.SCALE_GRANULARITY_N)
    k_groups = _ceil_div(k, base.SCALE_GRANULARITY_K)
    n_padded = n_groups * base.SCALE_GRANULARITY_N
    k_padded = k_groups * base.SCALE_GRANULARITY_K
    x_padded = F.pad(x_2d.float(), (0, k_padded - k, 0, n_padded - n))
    x_blocks = x_padded.view(
        n_groups,
        base.SCALE_GRANULARITY_N,
        k_groups,
        base.SCALE_GRANULARITY_K,
    ).permute(0, 2, 1, 3)
    logical_scale = _scale_from_amax(
        x_blocks.abs().amax(dim=(2, 3), keepdim=True)
    ).to(torch.float32)
    q_blocks = (x_blocks / logical_scale).to(torch.float8_e4m3fn)
    q_logical = q_blocks.permute(0, 2, 1, 3).reshape(n_padded, k_padded)[:, :k].contiguous()
    logical_scale = logical_scale.squeeze(-1).squeeze(-1).unsqueeze(-1).contiguous()
    return q_logical, logical_scale, logical_scale


def quantize_cutlass_groupwise_a(x_2d: Tensor) -> BlockScaledQuantizedOperand:
    q, logical_scale, kernel_scale = _quantize_cutlass_groupwise_a_tensors(x_2d)
    return BlockScaledQuantizedOperand(q, logical_scale, kernel_scale)


def quantize_cutlass_groupwise_b(x_2d: Tensor) -> BlockScaledQuantizedOperand:
    q, logical_scale, kernel_scale = _quantize_cutlass_groupwise_b_tensors(x_2d)
    return BlockScaledQuantizedOperand(q, logical_scale, kernel_scale)


@lru_cache(maxsize=1)
def _get_compiled_cutlass_quantizers():
    return (
        torch.compile(_quantize_cutlass_groupwise_a_tensors, dynamic=False, fullgraph=True),
        torch.compile(_quantize_cutlass_groupwise_b_tensors, dynamic=False, fullgraph=True),
    )


class CutlassGroupwiseGemmRunner:
    def __init__(
        self,
        raster_order: int = 0,
        swizzle_size: int = 1,
        *,
        autotune: bool = True,
        compile_helpers: bool = False,
        autotune_warmup: int = 3,
        autotune_iterations: int = 10,
    ):
        _load_cutlass_groupwise_custom_op()
        self.raster_order = raster_order
        self.swizzle_size = swizzle_size
        self.autotune = autotune
        self.compile_helpers = compile_helpers
        self.autotune_warmup = autotune_warmup
        self.autotune_iterations = autotune_iterations
        self._op = torch.ops.fp8_cutlass_groupwise.gemm
        self._tuned_configs: dict[tuple[int, int, int], tuple[int, int]] = {}
        if compile_helpers:
            self._quantize_a_impl, self._quantize_b_impl = _get_compiled_cutlass_quantizers()
        else:
            self._quantize_a_impl = _quantize_cutlass_groupwise_a_tensors
            self._quantize_b_impl = _quantize_cutlass_groupwise_b_tensors

    @staticmethod
    def _signature(a: BlockScaledQuantizedOperand, b: BlockScaledQuantizedOperand) -> tuple[int, int, int]:
        return (a.data.shape[0], b.data.shape[0], a.data.shape[1])

    def quantize_a(self, x_2d: Tensor) -> BlockScaledQuantizedOperand:
        q, logical_scale, kernel_scale = self._quantize_a_impl(x_2d.contiguous())
        return BlockScaledQuantizedOperand(q, logical_scale, kernel_scale)

    def quantize_b(self, x_2d: Tensor) -> BlockScaledQuantizedOperand:
        q, logical_scale, kernel_scale = self._quantize_b_impl(x_2d.contiguous())
        return BlockScaledQuantizedOperand(q, logical_scale, kernel_scale)

    def _measure_quantized_config(
        self,
        a: BlockScaledQuantizedOperand,
        b: BlockScaledQuantizedOperand,
        raster_order: int,
        swizzle_size: int,
    ) -> float:
        op = self._op
        for _ in range(self.autotune_warmup):
            op(a.data, b.data, a.kernel_scale, b.kernel_scale, raster_order, swizzle_size)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.autotune_iterations):
            op(a.data, b.data, a.kernel_scale, b.kernel_scale, raster_order, swizzle_size)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / max(self.autotune_iterations, 1)

    def _select_config(
        self,
        a: BlockScaledQuantizedOperand,
        b: BlockScaledQuantizedOperand,
    ) -> tuple[int, int]:
        key = self._signature(a, b)
        cached = self._tuned_configs.get(key)
        if cached is not None:
            return cached
        best_config = (self.raster_order, self.swizzle_size)
        if self.autotune:
            best_ms = float("inf")
            for raster_order, swizzle_size in _CUTLASS_GROUPWISE_AUTOTUNE_CANDIDATES:
                ms = self._measure_quantized_config(a, b, raster_order, swizzle_size)
                if ms < best_ms:
                    best_ms = ms
                    best_config = (raster_order, swizzle_size)
        self._tuned_configs[key] = best_config
        return best_config

    def matmul_quantized(
        self,
        a: BlockScaledQuantizedOperand,
        b: BlockScaledQuantizedOperand,
        out_dtype: torch.dtype,
    ) -> Tensor:
        raster_order, swizzle_size = self._select_config(a, b)
        out = self._op(
            a.data,
            b.data,
            a.kernel_scale,
            b.kernel_scale,
            raster_order,
            swizzle_size,
        )
        return out if out.dtype == out_dtype else out.to(out_dtype)

    def matmul(
        self,
        a_2d: Tensor,
        b_2d: Tensor,
        out_dtype: torch.dtype | None = None,
    ) -> Tensor:
        qa = self.quantize_a(a_2d)
        qb = self.quantize_b(b_2d)
        return self.matmul_quantized(qa, qb, out_dtype or a_2d.dtype)


class CutlassGroupwiseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        runner: CutlassGroupwiseGemmRunner,
    ) -> Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = runner.matmul(
            x_2d,
            weight.contiguous(),
            out_dtype=x.dtype,
        )
        if bias is not None:
            y_2d = y_2d + bias.to(y_2d.dtype)
        ctx.runner = runner
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.weight_dtype = weight.dtype
        ctx.save_for_backward(x_2d, weight.contiguous())
        return y_2d.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, weight = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_input_2d = ctx.runner.matmul(
            grad_out_2d,
            weight.transpose(0, 1).contiguous(),
            out_dtype=ctx.input_dtype,
        )
        grad_weight = ctx.runner.matmul(
            grad_out_2d.transpose(0, 1).contiguous(),
            x_2d.transpose(0, 1).contiguous(),
            out_dtype=ctx.weight_dtype,
        )
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_out_2d.sum(dim=0).to(ctx.bias_dtype)
        return grad_input_2d.view(ctx.input_shape), grad_weight, grad_bias, None


class CutlassGroupwiseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        raster_order: int = 0,
        swizzle_size: int = 1,
        autotune: bool = True,
        compile_helpers: bool = False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        self.runner = CutlassGroupwiseGemmRunner(
            raster_order=raster_order,
            swizzle_size=swizzle_size,
            autotune=autotune,
            compile_helpers=compile_helpers,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        **runner_kwargs,
    ) -> "CutlassGroupwiseLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **runner_kwargs,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def forward(self, x: Tensor) -> Tensor:
        return CutlassGroupwiseLinearFunction.apply(x, self.weight, self.bias, self.runner)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class CutlassGroupwiseForwardFp8BackwardBf16LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        runner: CutlassGroupwiseGemmRunner,
    ) -> Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y_2d = runner.matmul(
            x_2d,
            weight.contiguous(),
            out_dtype=x.dtype,
        )
        if bias is not None:
            y_2d = y_2d + bias.to(y_2d.dtype)
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.save_for_backward(x_2d, weight.contiguous())
        return y_2d.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_2d, weight = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_input_2d = torch.mm(grad_out_2d, weight).to(ctx.input_dtype)
        grad_weight = torch.mm(
            grad_out_2d.transpose(0, 1),
            x_2d,
        ).to(ctx.weight_dtype)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_out_2d.sum(dim=0).to(ctx.bias_dtype)
        return grad_input_2d.view(ctx.input_shape), grad_weight, grad_bias, None


class CutlassGroupwiseForwardFp8BackwardBf16Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        raster_order: int = 0,
        swizzle_size: int = 1,
        autotune: bool = True,
        compile_helpers: bool = False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        self.runner = CutlassGroupwiseGemmRunner(
            raster_order=raster_order,
            swizzle_size=swizzle_size,
            autotune=autotune,
            compile_helpers=compile_helpers,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        **runner_kwargs,
    ) -> "CutlassGroupwiseForwardFp8BackwardBf16Linear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            **runner_kwargs,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def forward(self, x: Tensor) -> Tensor:
        return CutlassGroupwiseForwardFp8BackwardBf16LinearFunction.apply(
            x, self.weight, self.bias, self.runner
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


@lru_cache(maxsize=None)
def _import_train_gpt_module(train_gpt_path: str):
    path = Path(train_gpt_path).resolve()
    saved_env = {key: os.environ.get(key) for key in _TRAIN_GPT_ENV_KEYS}
    try:
        for key in _TRAIN_GPT_ENV_KEYS:
            os.environ.pop(key, None)
        spec = importlib.util.spec_from_file_location(
            f"track_fp8_train_gpt_{abs(hash(path))}", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def get_train_gpt_mlp_shapes(train_gpt_path: str | os.PathLike = DEFAULT_TRAIN_GPT_PATH) -> dict:
    module = _import_train_gpt_module(str(train_gpt_path))
    h = module.Hyperparameters()
    device_tokens = h.train_batch_tokens // (h.world_size * h.grad_accum_steps)
    device_batch_size = device_tokens // h.train_seq_len
    microbatch_tokens = device_batch_size * h.train_seq_len
    hidden = int(h.mlp_mult * h.model_dim)
    return {
        "batch_size": device_batch_size,
        "seq_len": h.train_seq_len,
        "microbatch_tokens": microbatch_tokens,
        "model_dim": h.model_dim,
        "hidden_dim": hidden,
        "fc": (microbatch_tokens, h.model_dim, hidden),
        "proj": (microbatch_tokens, hidden, h.model_dim),
    }


def _compile_casted_linear_baseline(module: nn.Module):
    return torch.compile(module, dynamic=False, fullgraph=True)


def _forward_ms(forward_fn, x: Tensor, warmup: int, iterations: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            forward_fn(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            forward_fn(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iterations, 1)


def _backward_ms(forward_fn, module: nn.Module, x: Tensor, grad_out: Tensor, warmup: int, iterations: int) -> float:
    x_leaf = x.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        x_leaf.grad = None
        y = forward_fn(x_leaf)
        y.backward(grad_out)
    torch.cuda.synchronize()
    total_ms = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        module.zero_grad(set_to_none=True)
        x_leaf.grad = None
        y = forward_fn(x_leaf)
        torch.cuda.synchronize()
        start.record()
        y.backward(grad_out)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / max(iterations, 1)


def _forward_backward_ms(
    forward_fn,
    module: nn.Module,
    x: Tensor,
    grad_out: Tensor,
    warmup: int,
    iterations: int,
) -> float:
    x_leaf = x.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        x_leaf.grad = None
        y = forward_fn(x_leaf)
        y.backward(grad_out)
    torch.cuda.synchronize()
    total_ms = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        module.zero_grad(set_to_none=True)
        x_leaf.grad = None
        torch.cuda.synchronize()
        start.record()
        y = forward_fn(x_leaf)
        y.backward(grad_out)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / max(iterations, 1)


def _run_linear_once(forward_fn, module: nn.Module, x: Tensor, grad_out: Tensor):
    x_leaf = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    y = forward_fn(x_leaf)
    y.backward(grad_out)
    grad_bias = None if module.bias is None else module.bias.grad.detach().clone()
    return (
        y.detach().clone(),
        x_leaf.grad.detach().clone(),
        module.weight.grad.detach().clone(),
        grad_bias,
    )


def _run_linear_reference(weight: Tensor, bias: Tensor | None, x: Tensor, grad_out: Tensor):
    x_ref = x.detach().clone().float().requires_grad_(True)
    w_ref = weight.detach().clone().float().requires_grad_(True)
    b_ref = None
    if bias is not None:
        b_ref = bias.detach().clone().float().requires_grad_(True)
    y = F.linear(x_ref, w_ref, b_ref)
    y.backward(grad_out.float())
    grad_bias = None if b_ref is None else b_ref.grad.detach().clone()
    return (
        y.detach().clone(),
        x_ref.grad.detach().clone(),
        w_ref.grad.detach().clone(),
        grad_bias,
    )


def _run_module_once(forward_fn, module: nn.Module, x: Tensor, grad_out: Tensor):
    x_leaf = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    y = forward_fn(x_leaf)
    y.backward(grad_out)
    param_grads = {
        name: param.grad.detach().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    return y.detach().clone(), x_leaf.grad.detach().clone(), param_grads


def _tensor_error_stats(test: Tensor, ref: Tensor) -> dict:
    test_f = test.float()
    ref_f = ref.float()
    diff = test_f - ref_f
    denom = ref_f.norm().clamp_min(1e-8)
    return {
        "finite": bool(torch.isfinite(test_f).all().item()),
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "rel_l2": diff.norm().item() / denom.item(),
    }


def _summarize_trials(trials: list[dict]) -> dict:
    if not trials:
        return {}
    summary = {}
    for family in trials[0]:
        summary[family] = {}
        for tensor_name in trials[0][family]:
            family_trials = [trial[family][tensor_name] for trial in trials]
            summary[family][tensor_name] = {
                "all_finite": all(item["finite"] for item in family_trials),
                "worst_max_abs": max(item["max_abs"] for item in family_trials),
                "worst_mean_abs": max(item["mean_abs"] for item in family_trials),
                "worst_rmse": max(item["rmse"] for item in family_trials),
                "worst_rel_l2": max(item["rel_l2"] for item in family_trials),
            }
    return summary


def benchmark_against_train_gpt_casted_linear(
    train_gpt_path: str | os.PathLike = DEFAULT_TRAIN_GPT_PATH,
    *,
    warmup: int = 5,
    iterations: int = 20,
    stability_trials: int = 3,
    tile_shape_mn: tuple[int, int] = (128, 256),
    cluster_shape_mn: tuple[int, int] = (1, 1),
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the persistent blockscaled linear benchmark")
    module = _import_train_gpt_module(str(train_gpt_path))
    CastedLinear = module.CastedLinear
    shapes = get_train_gpt_mlp_shapes(train_gpt_path)
    batch_size = shapes["batch_size"]
    seq_len = shapes["seq_len"]
    results = {
        "train_gpt_path": str(Path(train_gpt_path).resolve()),
        "shape_summary": shapes,
        "cases": {},
    }
    for case_name, (_, in_features, out_features) in (
        ("fc", shapes["fc"]),
        ("proj", shapes["proj"]),
    ):
        torch.manual_seed(100 + out_features)
        baseline = CastedLinear(
            in_features,
            out_features,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        custom = PersistentBlockScaledLinear.from_linear(
            baseline,
            tile_shape_mn=tile_shape_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
        compiled_baseline = _compile_casted_linear_baseline(baseline)
        x = torch.randn(
            batch_size, seq_len, in_features, device="cuda", dtype=torch.bfloat16
        )
        grad_out = torch.randn(
            batch_size, seq_len, out_features, device="cuda", dtype=torch.bfloat16
        )
        forward_ms_baseline = _forward_ms(compiled_baseline, x, warmup, iterations)
        forward_ms_custom = _forward_ms(custom, x, warmup, iterations)
        backward_ms_baseline = _backward_ms(
            compiled_baseline, baseline, x, grad_out, warmup, iterations
        )
        backward_ms_custom = _backward_ms(
            custom, custom, x, grad_out, warmup, iterations
        )

        stability = []
        for seed in range(stability_trials):
            torch.manual_seed(1000 + seed + out_features)
            x_trial = torch.randn(
                batch_size, seq_len, in_features, device="cuda", dtype=torch.bfloat16
            )
            grad_trial = torch.randn(
                batch_size, seq_len, out_features, device="cuda", dtype=torch.bfloat16
            )
            out_custom, grad_input_custom, grad_weight_custom, _ = _run_linear_once(
                custom, custom, x_trial, grad_trial
            )
            out_baseline, grad_input_baseline, grad_weight_baseline, _ = _run_linear_once(
                compiled_baseline, baseline, x_trial, grad_trial
            )
            out_ref, grad_input_ref, grad_weight_ref, _ = _run_linear_reference(
                baseline.weight,
                baseline.bias,
                x_trial,
                grad_trial,
            )
            stability.append(
                {
                    "custom_vs_baseline": {
                        "forward": _tensor_error_stats(out_custom, out_baseline),
                        "grad_input": _tensor_error_stats(
                            grad_input_custom, grad_input_baseline
                        ),
                        "grad_weight": _tensor_error_stats(
                            grad_weight_custom, grad_weight_baseline
                        ),
                    },
                    "custom_vs_fp32": {
                        "forward": _tensor_error_stats(out_custom, out_ref),
                        "grad_input": _tensor_error_stats(grad_input_custom, grad_input_ref),
                        "grad_weight": _tensor_error_stats(
                            grad_weight_custom, grad_weight_ref
                        ),
                    },
                    "baseline_vs_fp32": {
                        "forward": _tensor_error_stats(out_baseline, out_ref),
                        "grad_input": _tensor_error_stats(
                            grad_input_baseline, grad_input_ref
                        ),
                        "grad_weight": _tensor_error_stats(
                            grad_weight_baseline, grad_weight_ref
                        ),
                    },
                }
            )
        results["cases"][case_name] = {
            "input_shape": (batch_size, seq_len, in_features),
            "output_shape": (batch_size, seq_len, out_features),
            "forward_ms": {
                "persistent_blockscaled": forward_ms_custom,
                "compiled_bf16_castedlinear": forward_ms_baseline,
            },
            "backward_ms": {
                "persistent_blockscaled": backward_ms_custom,
                "compiled_bf16_castedlinear": backward_ms_baseline,
            },
            "throughput_ratio": {
                "forward_vs_compiled_bf16": (
                    forward_ms_baseline / forward_ms_custom if forward_ms_custom > 0 else float("inf")
                ),
                "backward_vs_compiled_bf16": (
                    backward_ms_baseline / backward_ms_custom if backward_ms_custom > 0 else float("inf")
                ),
            },
            "stability_trials": stability,
            "stability_summary": _summarize_trials(stability),
        }
    return results


def benchmark_cutlass_custom_op_against_train_gpt_casted_linear(
    train_gpt_path: str | os.PathLike = DEFAULT_TRAIN_GPT_PATH,
    *,
    warmup: int = 5,
    iterations: int = 20,
    stability_trials: int = 3,
    autotune: bool = True,
    compile_helpers: bool = False,
    raster_order: int = 0,
    swizzle_size: int = 1,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CUTLASS blockscaled linear benchmark")
    _load_cutlass_groupwise_custom_op()
    module = _import_train_gpt_module(str(train_gpt_path))
    CastedLinear = module.CastedLinear
    shapes = get_train_gpt_mlp_shapes(train_gpt_path)
    batch_size = shapes["batch_size"]
    seq_len = shapes["seq_len"]
    results = {
        "train_gpt_path": str(Path(train_gpt_path).resolve()),
        "shape_summary": shapes,
        "custom_op": "fp8_cutlass_groupwise::gemm",
        "autotune": autotune,
        "compile_helpers": compile_helpers,
        "cases": {},
    }
    for case_name, (_, in_features, out_features) in (
        ("fc", shapes["fc"]),
        ("proj", shapes["proj"]),
    ):
        torch.manual_seed(700 + out_features)
        baseline = CastedLinear(
            in_features,
            out_features,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        custom = CutlassGroupwiseLinear.from_linear(
            baseline,
            raster_order=raster_order,
            swizzle_size=swizzle_size,
            autotune=autotune,
            compile_helpers=compile_helpers,
        )
        compiled_baseline = _compile_casted_linear_baseline(baseline)
        x = torch.randn(
            batch_size, seq_len, in_features, device="cuda", dtype=torch.bfloat16
        )
        grad_out = torch.randn(
            batch_size, seq_len, out_features, device="cuda", dtype=torch.bfloat16
        )
        forward_ms_baseline = _forward_ms(compiled_baseline, x, warmup, iterations)
        forward_ms_custom = _forward_ms(custom, x, warmup, iterations)
        backward_ms_baseline = _backward_ms(
            compiled_baseline, baseline, x, grad_out, warmup, iterations
        )
        backward_ms_custom = _backward_ms(
            custom, custom, x, grad_out, warmup, iterations
        )

        stability = []
        for seed in range(stability_trials):
            torch.manual_seed(1700 + seed + out_features)
            x_trial = torch.randn(
                batch_size, seq_len, in_features, device="cuda", dtype=torch.bfloat16
            )
            grad_trial = torch.randn(
                batch_size, seq_len, out_features, device="cuda", dtype=torch.bfloat16
            )
            out_custom, grad_input_custom, grad_weight_custom, _ = _run_linear_once(
                custom, custom, x_trial, grad_trial
            )
            out_baseline, grad_input_baseline, grad_weight_baseline, _ = _run_linear_once(
                compiled_baseline, baseline, x_trial, grad_trial
            )
            out_ref, grad_input_ref, grad_weight_ref, _ = _run_linear_reference(
                baseline.weight,
                baseline.bias,
                x_trial,
                grad_trial,
            )
            stability.append(
                {
                    "custom_vs_baseline": {
                        "forward": _tensor_error_stats(out_custom, out_baseline),
                        "grad_input": _tensor_error_stats(
                            grad_input_custom, grad_input_baseline
                        ),
                        "grad_weight": _tensor_error_stats(
                            grad_weight_custom, grad_weight_baseline
                        ),
                    },
                    "custom_vs_fp32": {
                        "forward": _tensor_error_stats(out_custom, out_ref),
                        "grad_input": _tensor_error_stats(grad_input_custom, grad_input_ref),
                        "grad_weight": _tensor_error_stats(
                            grad_weight_custom, grad_weight_ref
                        ),
                    },
                    "baseline_vs_fp32": {
                        "forward": _tensor_error_stats(out_baseline, out_ref),
                        "grad_input": _tensor_error_stats(
                            grad_input_baseline, grad_input_ref
                        ),
                        "grad_weight": _tensor_error_stats(
                            grad_weight_baseline, grad_weight_ref
                        ),
                    },
                }
            )

        tuned_configs = dict(custom.runner._tuned_configs)
        results["cases"][case_name] = {
            "input_shape": (batch_size, seq_len, in_features),
            "output_shape": (batch_size, seq_len, out_features),
            "forward_ms": {
                "cutlass_custom_op": forward_ms_custom,
                "compiled_bf16_castedlinear": forward_ms_baseline,
            },
            "backward_ms": {
                "cutlass_custom_op": backward_ms_custom,
                "compiled_bf16_castedlinear": backward_ms_baseline,
            },
            "throughput_ratio": {
                "forward_vs_compiled_bf16": (
                    forward_ms_baseline / forward_ms_custom if forward_ms_custom > 0 else float("inf")
                ),
                "backward_vs_compiled_bf16": (
                    backward_ms_baseline / backward_ms_custom if backward_ms_custom > 0 else float("inf")
                ),
            },
            "autotuned_quantized_gemm_configs": {
                f"{m}x{n}x{k}": {"raster_order": raster, "swizzle_size": swizzle}
                for (m, n, k), (raster, swizzle) in tuned_configs.items()
            },
            "stability_trials": stability,
            "stability_summary": _summarize_trials(stability),
        }
    return results


def _instantiate_train_gpt_mlp(train_gpt_path: str | os.PathLike, *, device: str = "cuda"):
    module = _import_train_gpt_module(str(train_gpt_path))
    h = module.Hyperparameters()
    mlp = module.MLP(h.model_dim, h.mlp_mult).to(device=device, dtype=torch.bfloat16)
    return module, h, mlp


def _replace_train_gpt_mlp_linears(
    mlp: nn.Module,
    linear_cls: type[nn.Module],
    **runner_kwargs,
) -> nn.Module:
    mlp.fc = linear_cls.from_linear(mlp.fc, **runner_kwargs)
    mlp.proj = linear_cls.from_linear(mlp.proj, **runner_kwargs)
    return mlp


def _run_train_gpt_mlp_reference(mlp_module: nn.Module, x: Tensor, grad_out: Tensor):
    x_ref = x.detach().clone().float().requires_grad_(True)
    fc_weight = mlp_module.fc.weight.detach().clone().float().requires_grad_(True)
    proj_weight = mlp_module.proj.weight.detach().clone().float().requires_grad_(True)
    fc_bias = None
    proj_bias = None
    if mlp_module.fc.bias is not None:
        fc_bias = mlp_module.fc.bias.detach().clone().float().requires_grad_(True)
    if mlp_module.proj.bias is not None:
        proj_bias = mlp_module.proj.bias.detach().clone().float().requires_grad_(True)
    hidden = F.linear(x_ref, fc_weight, fc_bias)
    hidden = F.leaky_relu(hidden, negative_slope=0.5).square()
    y = F.linear(hidden, proj_weight, proj_bias)
    y.backward(grad_out.float())
    grads = {
        "fc.weight": fc_weight.grad.detach().clone(),
        "proj.weight": proj_weight.grad.detach().clone(),
    }
    if fc_bias is not None:
        grads["fc.bias"] = fc_bias.grad.detach().clone()
    if proj_bias is not None:
        grads["proj.bias"] = proj_bias.grad.detach().clone()
    return y.detach().clone(), x_ref.grad.detach().clone(), grads


def benchmark_train_gpt_mlp_training_like(
    train_gpt_path: str | os.PathLike = DEFAULT_TRAIN_GPT_PATH,
    *,
    warmup: int = 5,
    iterations: int = 20,
    stability_trials: int = 3,
    compile_helpers: bool = True,
    autotune: bool = True,
    include_full_fp8_backward: bool = True,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the train-like MLP benchmark")
    _load_cutlass_groupwise_custom_op()
    module, h, baseline = _instantiate_train_gpt_mlp(train_gpt_path)
    shapes = get_train_gpt_mlp_shapes(train_gpt_path)
    x = torch.randn(
        shapes["batch_size"],
        shapes["seq_len"],
        h.model_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_out = torch.randn_like(x)

    compiled_baseline = _compile_casted_linear_baseline(baseline)

    hybrid = _replace_train_gpt_mlp_linears(
        module.MLP(h.model_dim, h.mlp_mult).to(device="cuda", dtype=torch.bfloat16),
        CutlassGroupwiseForwardFp8BackwardBf16Linear,
        compile_helpers=compile_helpers,
        autotune=autotune,
    )
    hybrid.load_state_dict(baseline.state_dict())

    variants: dict[str, tuple[object, nn.Module]] = {
        "compiled_bf16_castedlinear_mlp": (compiled_baseline, baseline),
        "cutlass_fp8_forward_bf16_backward_mlp": (hybrid, hybrid),
    }

    full_fp8 = None
    if include_full_fp8_backward:
        full_fp8 = _replace_train_gpt_mlp_linears(
            module.MLP(h.model_dim, h.mlp_mult).to(device="cuda", dtype=torch.bfloat16),
            CutlassGroupwiseLinear,
            compile_helpers=compile_helpers,
            autotune=autotune,
        )
        full_fp8.load_state_dict(baseline.state_dict())
        variants["cutlass_fp8_forward_fp8_backward_mlp"] = (full_fp8, full_fp8)

    results = {
        "train_gpt_path": str(Path(train_gpt_path).resolve()),
        "shape_summary": shapes,
        "module": "train_gpt.MLP",
        "compile_helpers": compile_helpers,
        "autotune": autotune,
        "variants": {},
    }

    for name, (forward_fn, module_obj) in variants.items():
        results["variants"][name] = {
            "forward_ms": _forward_ms(forward_fn, x, warmup, iterations),
            "backward_ms": _backward_ms(forward_fn, module_obj, x, grad_out, warmup, iterations),
            "forward_backward_ms": _forward_backward_ms(
                forward_fn, module_obj, x, grad_out, warmup, iterations
            ),
        }
        if hasattr(module_obj, "fc") and hasattr(module_obj.fc, "runner"):
            results["variants"][name]["fc_autotuned_configs"] = {
                f"{m}x{n}x{k}": {"raster_order": raster, "swizzle_size": swizzle}
                for (m, n, k), (raster, swizzle) in module_obj.fc.runner._tuned_configs.items()
            }
        if hasattr(module_obj, "proj") and hasattr(module_obj.proj, "runner"):
            results["variants"][name]["proj_autotuned_configs"] = {
                f"{m}x{n}x{k}": {"raster_order": raster, "swizzle_size": swizzle}
                for (m, n, k), (raster, swizzle) in module_obj.proj.runner._tuned_configs.items()
            }

    stability = []
    for seed in range(stability_trials):
        torch.manual_seed(3100 + seed)
        x_trial = torch.randn(
            shapes["batch_size"],
            shapes["seq_len"],
            h.model_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad_trial = torch.randn_like(x_trial)

        baseline_out, baseline_grad_input, baseline_param_grads = _run_module_once(
            compiled_baseline,
            baseline,
            x_trial,
            grad_trial,
        )
        ref_out, ref_grad_input, ref_param_grads = _run_train_gpt_mlp_reference(
            baseline,
            x_trial,
            grad_trial,
        )

        trial = {
            "baseline_vs_fp32": {
                "forward": _tensor_error_stats(baseline_out, ref_out),
                "grad_input": _tensor_error_stats(baseline_grad_input, ref_grad_input),
                "fc.weight": _tensor_error_stats(
                    baseline_param_grads["fc.weight"], ref_param_grads["fc.weight"]
                ),
                "proj.weight": _tensor_error_stats(
                    baseline_param_grads["proj.weight"], ref_param_grads["proj.weight"]
                ),
            }
        }

        hybrid_out, hybrid_grad_input, hybrid_param_grads = _run_module_once(
            hybrid,
            hybrid,
            x_trial,
            grad_trial,
        )
        trial["hybrid_vs_baseline"] = {
            "forward": _tensor_error_stats(hybrid_out, baseline_out),
            "grad_input": _tensor_error_stats(hybrid_grad_input, baseline_grad_input),
            "fc.weight": _tensor_error_stats(
                hybrid_param_grads["fc.weight"], baseline_param_grads["fc.weight"]
            ),
            "proj.weight": _tensor_error_stats(
                hybrid_param_grads["proj.weight"], baseline_param_grads["proj.weight"]
            ),
        }
        trial["hybrid_vs_fp32"] = {
            "forward": _tensor_error_stats(hybrid_out, ref_out),
            "grad_input": _tensor_error_stats(hybrid_grad_input, ref_grad_input),
            "fc.weight": _tensor_error_stats(
                hybrid_param_grads["fc.weight"], ref_param_grads["fc.weight"]
            ),
            "proj.weight": _tensor_error_stats(
                hybrid_param_grads["proj.weight"], ref_param_grads["proj.weight"]
            ),
        }

        if full_fp8 is not None:
            full_out, full_grad_input, full_param_grads = _run_module_once(
                full_fp8,
                full_fp8,
                x_trial,
                grad_trial,
            )
            trial["full_fp8_vs_baseline"] = {
                "forward": _tensor_error_stats(full_out, baseline_out),
                "grad_input": _tensor_error_stats(full_grad_input, baseline_grad_input),
                "fc.weight": _tensor_error_stats(
                    full_param_grads["fc.weight"], baseline_param_grads["fc.weight"]
                ),
                "proj.weight": _tensor_error_stats(
                    full_param_grads["proj.weight"], baseline_param_grads["proj.weight"]
                ),
            }
            trial["full_fp8_vs_fp32"] = {
                "forward": _tensor_error_stats(full_out, ref_out),
                "grad_input": _tensor_error_stats(full_grad_input, ref_grad_input),
                "fc.weight": _tensor_error_stats(
                    full_param_grads["fc.weight"], ref_param_grads["fc.weight"]
                ),
                "proj.weight": _tensor_error_stats(
                    full_param_grads["proj.weight"], ref_param_grads["proj.weight"]
                ),
            }

        stability.append(trial)

    results["stability_trials"] = stability
    results["stability_summary"] = _summarize_trials(stability)
    return results


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
