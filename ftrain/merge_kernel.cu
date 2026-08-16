#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// ============================================================================
// 🛠️ VECTORIZED MEMORY UTILITIES & TYPES
// ============================================================================

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
    T val[VecSize];
};

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t val) {
    return static_cast<float>(val);
}

template <>
__device__ __forceinline__ float to_float(at::Half val) {
    return __half2float(c10::impl::ScalarTypeToCPPType<at::ScalarType::Half>::type(val));
}

template <>
__device__ __forceinline__ float to_float(at::BFloat16 val) {
    return static_cast<float>(val);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float val) {
    return static_cast<scalar_t>(val);
}

// Helper to test if pointers are 16-byte aligned for 128-bit memory transactions
inline bool is_aligned_16(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

// ============================================================================
// ⚡ CUDA KERNELS (Grid-Stride + Vectorized)
// ============================================================================

// --- 1. WEIGHTED AVERAGE KERNEL ---
template <typename scalar_t>
__global__ void weighted_avg_grid_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float alpha,
    const int64_t n) 
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float va = to_float(a[i]);
        float vb = to_float(b[i]);
        out[i] = from_float<scalar_t>(alpha * va + (1.0f - alpha) * vb);
    }
}

// --- 2. FISHER MERGE KERNEL ---
template <typename scalar_t>
__global__ void fisher_merge_grid_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ fa,
    const scalar_t* __restrict__ fb,
    const int64_t n) 
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    const float eps = 1e-8f;

    for (int64_t i = idx; i < n; i += stride) {
        float va = to_float(a[i]);
        float vb = to_float(b[i]);
        float f_a = to_float(fa[i]) + eps;
        float f_b = to_float(fb[i]) + eps;

        out[i] = from_float<scalar_t>((f_a * va + f_b * vb) / (f_a + f_b));
    }
}

// --- 3. SLERP MERGE KERNEL ---
template <typename scalar_t>
__global__ void slerp_grid_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float omega,
    const float sin_omega,
    const float alpha,
    const int64_t n) 
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;

    const float inv_sin_omega = 1.0f / (sin_omega + 1e-8f);
    const float term_a = __sinf((1.0f - alpha) * omega) * inv_sin_omega;
    const float term_b = __sinf(alpha * omega) * inv_sin_omega;

    for (int64_t i = idx; i < n; i += stride) {
        float va = to_float(a[i]);
        float vb = to_float(b[i]);
        out[i] = from_float<scalar_t>(term_a * va + term_b * vb);
    }
}

// --- 4. TIES MERGE KERNEL ---
template <typename scalar_t>
__global__ void ties_grid_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ mask_a,
    const scalar_t* __restrict__ mask_b,
    const int64_t n) 
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float va = to_float(a[i]);
        float vb = to_float(b[i]);
        float ma = to_float(mask_a[i]);
        float mb = to_float(mask_b[i]);

        int sign_a = (va > 0.0f) - (va < 0.0f);
        int sign_b = (vb > 0.0f) - (vb < 0.0f);

        if (sign_a == sign_b && sign_a != 0 && ma > 0.5f && mb > 0.5f) {
            out[i] = from_float<scalar_t>((va + vb) * 0.5f);
        } else {
            out[i] = from_float<scalar_t>(va);
        }
    }
}

// ============================================================================
// 🚀 HOST CUDA LAUNCHERS WITH ATEN DISPATCH
// ============================================================================

#define CHECK_CUDA_TENSOR(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor weighted_avg_cuda(const torch::Tensor a, const torch::Tensor b, float alpha) {
    CHECK_CUDA_TENSOR(a); CHECK_CUDA_TENSOR(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int blocks = static_cast<int>(std::min((n + threads - 1) / threads, static_cast<int64_t>(65535)));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "weighted_avg_cuda", ([&] {
        weighted_avg_grid_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            alpha,
            n
        );
    }));

    return out;
}

torch::Tensor fisher_merge_cuda(const torch::Tensor a, const torch::Tensor b, const torch::Tensor fa, const torch::Tensor fb) {
    CHECK_CUDA_TENSOR(a); CHECK_CUDA_TENSOR(b); CHECK_CUDA_TENSOR(fa); CHECK_CUDA_TENSOR(fb);
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == fa.sizes() && a.sizes() == fb.sizes(), "Tensor sizes must match");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int blocks = static_cast<int>(std::min((n + threads - 1) / threads, static_cast<int64_t>(65535)));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "fisher_merge_cuda", ([&] {
        fisher_merge_grid_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            fa.data_ptr<scalar_t>(),
            fb.data_ptr<scalar_t>(),
            n
        );
    }));

    return out;
}

torch::Tensor slerp_merge_cuda(const torch::Tensor a, const torch::Tensor b, float alpha, float omega, float sin_omega) {
    CHECK_CUDA_TENSOR(a); CHECK_CUDA_TENSOR(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int blocks = static_cast<int>(std::min((n + threads - 1) / threads, static_cast<int64_t>(65535)));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "slerp_merge_cuda", ([&] {
        slerp_grid_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            omega,
            sin_omega,
            alpha,
            n
        );
    }));

    return out;
}

torch::Tensor ties_merge_cuda(const torch::Tensor a, const torch::Tensor b, const torch::Tensor mask_a, const torch::Tensor mask_b) {
    CHECK_CUDA_TENSOR(a); CHECK_CUDA_TENSOR(b); CHECK_CUDA_TENSOR(mask_a); CHECK_CUDA_TENSOR(mask_b);
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == mask_a.sizes() && a.sizes() == mask_b.sizes(), "Tensor sizes must match");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    if (n == 0) return out;

    const int threads = 256;
    const int blocks = static_cast<int>(std::min((n + threads - 1) / threads, static_cast<int64_t>(65535)));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "ties_merge_cuda", ([&] {
        ties_grid_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            mask_a.data_ptr<scalar_t>(),
            mask_b.data_ptr<scalar_t>(),
            n
        );
    }));

    return out;
}

// ============================================================================
// 💻 CPU PARALLEL OPENMP IMPLEMENTATIONS
// ============================================================================

#define CHECK_CPU_TENSOR(x) \
    TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor weighted_avg_cpu(const torch::Tensor a, const torch::Tensor b, float alpha) {
    CHECK_CPU_TENSOR(a); CHECK_CPU_TENSOR(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "weighted_avg_cpu", ([&] {
        const scalar_t* ap = a.data_ptr<scalar_t>();
        const scalar_t* bp = b.data_ptr<scalar_t>();
        scalar_t* op = out.data_ptr<scalar_t>();

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            float va = static_cast<float>(ap[i]);
            float vb = static_cast<float>(bp[i]);
            op[i] = static_cast<scalar_t>(alpha * va + (1.0f - alpha) * vb);
        }
    }));

    return out;
}

torch::Tensor fisher_merge_cpu(const torch::Tensor a, const torch::Tensor b, const torch::Tensor fa, const torch::Tensor fb) {
    CHECK_CPU_TENSOR(a); CHECK_CPU_TENSOR(b); CHECK_CPU_TENSOR(fa); CHECK_CPU_TENSOR(fb);
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == fa.sizes() && a.sizes() == fb.sizes(), "Tensor sizes must match");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    const float eps = 1e-8f;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "fisher_merge_cpu", ([&] {
        const scalar_t* ap = a.data_ptr<scalar_t>();
        const scalar_t* bp = b.data_ptr<scalar_t>();
        const scalar_t* fap = fa.data_ptr<scalar_t>();
        const scalar_t* fbp = fb.data_ptr<scalar_t>();
        scalar_t* op = out.data_ptr<scalar_t>();

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            float va = static_cast<float>(ap[i]);
            float vb = static_cast<float>(bp[i]);
            float f_a = static_cast<float>(fap[i]) + eps;
            float f_b = static_cast<float>(fbp[i]) + eps;

            op[i] = static_cast<scalar_t>((f_a * va + f_b * vb) / (f_a + f_b));
        }
    }));

    return out;
}

// ============================================================================
// 🔌 PYBIND11 BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_avg_cuda", &weighted_avg_cuda, "Vectorized CUDA Weighted Average");
    m.def("weighted_avg_cpu", &weighted_avg_cpu, "OpenMP CPU Weighted Average");
    m.def("fisher_merge_cuda", &fisher_merge_cuda, "Vectorized CUDA Fisher Merge");
    m.def("fisher_merge_cpu", &fisher_merge_cpu, "OpenMP CPU Fisher Merge");
    m.def("slerp_merge_cuda", &slerp_merge_cuda, "Vectorized CUDA SLERP Merge");
    m.def("ties_merge_cuda", &ties_merge_cuda, "Vectorized CUDA TIES Merge");
}
