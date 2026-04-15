#include <torch/extension.h>

at::Tensor cutlass_groupwise_gemm_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& sfa,
    const at::Tensor& sfb,
    int64_t raster_order,
    int64_t swizzle_size);

TORCH_LIBRARY(fp8_cutlass_groupwise, m) {
  m.def(
      "gemm(Tensor a, Tensor b, Tensor sfa, Tensor sfb, int raster_order=0, int swizzle_size=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(fp8_cutlass_groupwise, CUDA, m) {
  m.impl("gemm", &cutlass_groupwise_gemm_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS FP8 groupwise GEMM custom op";
}
