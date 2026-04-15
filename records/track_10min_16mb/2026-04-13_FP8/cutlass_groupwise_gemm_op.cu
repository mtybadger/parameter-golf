#include <torch/extension.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>

#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace {

using namespace cute;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = void;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 1;

using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementBlockScale = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_256, _128, _128>;
using ClusterShape = Shape<_1, _2, _1>;

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;

using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
    ScaleGranularityM,
    ScaleGranularityN,
    ScaleGranularityK,
    cute::GMMA::Major::MN,
    cute::GMMA::Major::K>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueTileType = Shape<_128, _128>;
using FusionOperation = cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementD,
    LayoutD,
    AlignmentD,
    EpilogueSchedule,
    FusionOperation>::CollectiveOp;

using CollectiveMainloopWithBlockWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    cute::tuple<LayoutA, LayoutSFA>,
    AlignmentA,
    ElementB,
    cute::tuple<LayoutB, LayoutSFB>,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopWithBlockWiseScaling,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

constexpr int64_t ceil_div_int(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

void check_cutlass(cutlass::Status status, const char* context) {
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      context,
      " failed with CUTLASS status ",
      cutlassGetStatusString(status));
}

RasterOrderOptions raster_order_from_int(int64_t raster_order) {
  switch (raster_order) {
    case 1:
      return RasterOrderOptions::AlongM;
    case 2:
      return RasterOrderOptions::AlongN;
    default:
      return RasterOrderOptions::Heuristic;
  }
}

struct RunnerKey {
  int device = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t raster_order = 0;
  int64_t swizzle_size = 1;

  bool operator==(const RunnerKey& other) const {
    return device == other.device && m == other.m && n == other.n && k == other.k &&
        raster_order == other.raster_order && swizzle_size == other.swizzle_size;
  }
};

struct RunnerKeyHash {
  size_t operator()(const RunnerKey& key) const {
    size_t h = static_cast<size_t>(key.device);
    h = h * 1315423911u + static_cast<size_t>(key.m);
    h = h * 1315423911u + static_cast<size_t>(key.n);
    h = h * 1315423911u + static_cast<size_t>(key.k);
    h = h * 1315423911u + static_cast<size_t>(key.raster_order);
    h = h * 1315423911u + static_cast<size_t>(key.swizzle_size);
    return h;
  }
};

class CachedGemmRunner {
 public:
  explicit CachedGemmRunner(const RunnerKey& key)
      : key_(key),
        stride_a_(cutlass::make_cute_packed_stride(
            StrideA{},
            cute::make_shape(static_cast<int>(key.m), static_cast<int>(key.k), 1))),
        stride_b_(cutlass::make_cute_packed_stride(
            StrideB{},
            cute::make_shape(static_cast<int>(key.n), static_cast<int>(key.k), 1))),
        stride_c_(cutlass::make_cute_packed_stride(
            StrideC{},
            cute::make_shape(static_cast<int>(key.m), static_cast<int>(key.n), 1))),
        stride_d_(cutlass::make_cute_packed_stride(
            StrideD{},
            cute::make_shape(static_cast<int>(key.m), static_cast<int>(key.n), 1))),
        layout_sfa_(ScaleConfig::tile_atom_to_shape_SFA(make_shape(
            static_cast<int>(key.m),
            static_cast<int>(key.n),
            static_cast<int>(key.k),
            1))),
        layout_sfb_(ScaleConfig::tile_atom_to_shape_SFB(make_shape(
            static_cast<int>(key.m),
            static_cast<int>(key.n),
            static_cast<int>(key.k),
            1))) {}

  void run(
      const at::Tensor& a,
      const at::Tensor& b,
      const at::Tensor& sfa,
      const at::Tensor& sfb,
      at::Tensor& d) {
    std::lock_guard<std::mutex> lock(mutex_);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {static_cast<int>(key_.m), static_cast<int>(key_.n), static_cast<int>(key_.k), 1},
        {reinterpret_cast<ElementA*>(a.data_ptr()),
         stride_a_,
         reinterpret_cast<ElementB*>(b.data_ptr()),
         stride_b_,
         reinterpret_cast<ElementBlockScale*>(sfa.data_ptr()),
         layout_sfa_,
         reinterpret_cast<ElementBlockScale*>(sfb.data_ptr()),
         layout_sfb_},
        {{}, nullptr, stride_c_, reinterpret_cast<ElementD*>(d.data_ptr()), stride_d_}};

    args.scheduler.raster_order = raster_order_from_int(key_.raster_order);
    args.scheduler.max_swizzle_size = static_cast<int>(key_.swizzle_size);

    auto stream = at::cuda::getCurrentCUDAStream(key_.device).stream();
    if (!initialized_) {
      check_cutlass(Gemm::can_implement(args), "Gemm::can_implement");
      auto workspace_size = static_cast<int64_t>(Gemm::get_workspace_size(args));
      workspace_ = torch::empty(
          {workspace_size},
          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, key_.device));
      check_cutlass(
          gemm_.initialize(args, workspace_.data_ptr(), stream),
          "Gemm::initialize");
      initialized_ = true;
    } else {
      check_cutlass(gemm_.update(args), "Gemm::update");
    }

    check_cutlass(gemm_.run(stream), "Gemm::run");
    C10_CUDA_CHECK(cudaGetLastError());
  }

 private:
  RunnerKey key_;
  Gemm gemm_;
  at::Tensor workspace_;
  bool initialized_ = false;
  StrideA stride_a_;
  StrideB stride_b_;
  StrideC stride_c_;
  StrideD stride_d_;
  LayoutSFA layout_sfa_;
  LayoutSFB layout_sfb_;
  std::mutex mutex_;
};

std::shared_ptr<CachedGemmRunner> get_runner(const RunnerKey& key) {
  static std::mutex cache_mutex;
  static std::unordered_map<RunnerKey, std::shared_ptr<CachedGemmRunner>, RunnerKeyHash> cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  auto runner = std::make_shared<CachedGemmRunner>(key);
  cache.emplace(key, runner);
  return runner;
}

void validate_inputs(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& sfa,
    const at::Tensor& sfb,
    int64_t swizzle_size) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(sfa.is_cuda(), "sfa must be a CUDA tensor");
  TORCH_CHECK(sfb.is_cuda(), "sfb must be a CUDA tensor");
  TORCH_CHECK(
      a.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "a must use torch.float8_e4m3fn");
  TORCH_CHECK(
      b.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "b must use torch.float8_e4m3fn");
  TORCH_CHECK(sfa.scalar_type() == at::ScalarType::Float, "sfa must use torch.float32");
  TORCH_CHECK(sfb.scalar_type() == at::ScalarType::Float, "sfb must use torch.float32");
  TORCH_CHECK(a.dim() == 2, "a must be rank-2");
  TORCH_CHECK(b.dim() == 2, "b must be rank-2 and stored as [n, k]");
  TORCH_CHECK(sfa.dim() == 3 && sfa.size(2) == 1, "sfa must have shape [m, ceil_div(k,128), 1]");
  TORCH_CHECK(sfb.dim() == 3 && sfb.size(2) == 1, "sfb must have shape [ceil_div(n,128), ceil_div(k,128), 1]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous row-major");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous row-major [n, k]");
  TORCH_CHECK(sfb.is_contiguous(), "sfb must be contiguous");
  TORCH_CHECK(swizzle_size == 1 || swizzle_size == 2 || swizzle_size == 4 || swizzle_size == 8,
      "swizzle_size must be one of {1, 2, 4, 8}");

  auto props = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(props->major >= 9, "This kernel requires Hopper-class CUDA devices");

  TORCH_CHECK(a.get_device() == b.get_device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.get_device() == sfa.get_device(), "a and sfa must be on the same CUDA device");
  TORCH_CHECK(a.get_device() == sfb.get_device(), "a and sfb must be on the same CUDA device");

  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  TORCH_CHECK(b.size(1) == k, "b must have shape [n, k] with b.size(1) == a.size(1)");
  const int64_t n = b.size(0);
  const int64_t k_groups = ceil_div_int(k, ScaleGranularityK);
  const int64_t n_groups = ceil_div_int(n, ScaleGranularityN);

  TORCH_CHECK(sfa.size(0) == m, "sfa size(0) must match a.size(0)");
  TORCH_CHECK(sfa.size(1) == k_groups, "sfa size(1) must equal ceil_div(k, 128)");
  TORCH_CHECK(sfa.stride(0) == 1, "sfa must be packed with stride(0) == 1");
  TORCH_CHECK(sfa.stride(1) == m, "sfa must be packed with stride(1) == m");
  TORCH_CHECK(sfa.stride(2) == m * k_groups, "sfa must be packed with Hopper SFA stride");

  TORCH_CHECK(sfb.size(0) == n_groups, "sfb size(0) must equal ceil_div(n, 128)");
  TORCH_CHECK(sfb.size(1) == k_groups, "sfb size(1) must equal ceil_div(k, 128)");
}

}  // namespace

at::Tensor cutlass_groupwise_gemm_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& sfa,
    const at::Tensor& sfb,
    int64_t raster_order,
    int64_t swizzle_size) {
  validate_inputs(a, b, sfa, sfb, swizzle_size);
  at::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(a.get_device()));

  const auto m = a.size(0);
  const auto k = a.size(1);
  const auto n = b.size(0);

  at::Tensor d = torch::empty(
      {m, n},
      torch::TensorOptions().dtype(torch::kBFloat16).device(a.device()));

  RunnerKey key{
      a.get_device(),
      m,
      n,
      k,
      raster_order,
      swizzle_size,
  };
  auto runner = get_runner(key);
  runner->run(a, b, sfa, sfb, d);
  return d;
}
