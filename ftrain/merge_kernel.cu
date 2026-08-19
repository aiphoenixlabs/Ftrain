// ============================================================================
// 🔥 FTRAIN MERGE KERNEL 
// ============================================================================
//
// High-performance numerical kernels used by the FTRAIN intelligent merger.
//
// Supported operations
// --------------------
//   • Weighted average
//   • Fisher-weighted merge
//   • SLERP
//   • TIES
//
// Backends
// --------
//   • CUDA
//   • CPU / OpenMP
//
// Design goals
// ------------
//   • Production-grade validation
//   • Correct CUDA stream handling
//   • Explicit dtype/device validation
//   • Safe empty-tensor handling
//   • CUDA launch error checking
//   • CPU fallbacks for every exposed operation
//   • FP32 accumulation for low-precision tensors
//   • No silent broadcasting
//   • No hidden device transfers
//   • Compatible with the Python FTRAIN fast merge API
//
// IMPORTANT
// ---------
// These kernels operate on tensors with compatible shapes.
// Cross-architecture model compatibility must be handled by the higher-level
// merger/planner before these kernels are called.
//
// ============================================================================

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>


// ============================================================================
// CONFIGURATION
// ============================================================================

namespace ftrain {

constexpr int DEFAULT_THREADS = 256;
constexpr int MAX_GRID_BLOCKS = 65535;
constexpr float EPSILON = 1e-8f;


// ============================================================================
// VALIDATION HELPERS
// ============================================================================

inline void check_floating_dtype(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32 ||
        tensor.scalar_type() == torch::kFloat64 ||
        tensor.scalar_type() == torch::kFloat16 ||
        tensor.scalar_type() == torch::kBFloat16,
        name,
        " must use float32, float64, float16, or bfloat16. Got ",
        tensor.scalar_type()
    );
}


inline void check_cpu_tensor(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(
        tensor.defined(),
        name,
        " must be defined"
    );

    TORCH_CHECK(
        tensor.device().is_cpu(),
        name,
        " must be a CPU tensor. Got ",
        tensor.device()
    );

    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous"
    );

    check_floating_dtype(tensor, name);
}


inline void check_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(
        tensor.defined(),
        name,
        " must be defined"
    );

    TORCH_CHECK(
        tensor.device().is_cuda(),
        name,
        " must be a CUDA tensor. Got ",
        tensor.device()
    );

    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous"
    );

    check_floating_dtype(tensor, name);
}


inline void check_same_shape(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const char* operation
) {
    TORCH_CHECK(
        a.sizes() == b.sizes(),
        operation,
        ": tensor shape mismatch: ",
        a.sizes(),
        " vs ",
        b.sizes()
    );
}


inline void check_same_device(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const char* operation
) {
    TORCH_CHECK(
        a.device() == b.device(),
        operation,
        ": tensors must be on the same device. Got ",
        a.device(),
        " and ",
        b.device()
    );
}


inline void check_same_dtype(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const char* operation
) {
    TORCH_CHECK(
        a.scalar_type() == b.scalar_type(),
        operation,
        ": tensors must use the same dtype. Got ",
        a.scalar_type(),
        " and ",
        b.scalar_type()
    );
}


inline void validate_alpha(
    float alpha,
    const char* operation
) {
    TORCH_CHECK(
        std::isfinite(alpha),
        operation,
        ": alpha must be finite"
    );

    TORCH_CHECK(
        alpha >= 0.0f && alpha <= 1.0f,
        operation,
        ": alpha must be in [0, 1]. Got ",
        alpha
    );
}


inline void validate_merge_pair(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const char* operation
) {
    check_same_shape(a, b, operation);
    check_same_device(a, b, operation);
    check_same_dtype(a, b, operation);
}


// ============================================================================
// CUDA UTILITIES
// ============================================================================

inline int64_t calculate_blocks(int64_t n) {
    if (n <= 0) {
        return 1;
    }

    const int64_t blocks =
        (n + DEFAULT_THREADS - 1) / DEFAULT_THREADS;

    return std::min<int64_t>(
        blocks,
        MAX_GRID_BLOCKS
    );
}


inline cudaStream_t current_cuda_stream(
    const torch::Tensor& tensor
) {
    c10::cuda::CUDAGuard guard(tensor.device());

    return c10::cuda::getDefaultCUDAStream(
        tensor.device().index()
    );
}


// Use the actual PyTorch current stream.
//
// This helper exists separately because using a default CUDA stream for an
// operation invoked from a non-default PyTorch stream can introduce ordering
// bugs.
inline cudaStream_t pytorch_current_stream(
    const torch::Tensor& tensor
) {
    c10::cuda::CUDAGuard guard(tensor.device());

    return c10::cuda::getCurrentCUDAStream(
        tensor.device().index()
    ).stream();
}


inline void check_cuda_launch(
    const char* operation
) {
    const cudaError_t error = cudaGetLastError();

    TORCH_CHECK(
        error == cudaSuccess,
        operation,
        ": CUDA kernel launch failed: ",
        cudaGetErrorString(error)
    );
}


// ============================================================================
// DEVICE NUMERICAL UTILITIES
// ============================================================================

template <typename scalar_t>
__device__ __forceinline__ float to_float(
    scalar_t value
) {
    return static_cast<float>(value);
}


template <>
__device__ __forceinline__ float to_float<at::Half>(
    at::Half value
) {
    return __half2float(
        value
    );
}


template <>
__device__ __forceinline__ float to_float<at::BFloat16>(
    at::BFloat16 value
) {
    return static_cast<float>(value);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(
    float value
) {
    return static_cast<scalar_t>(value);
}


// ============================================================================
// CUDA KERNEL: WEIGHTED AVERAGE
// ============================================================================

template <typename scalar_t>
__global__ void weighted_avg_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    float alpha,
    int64_t n
) {
    const int64_t thread_id =
        static_cast<int64_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const int64_t stride =
        static_cast<int64_t>(blockDim.x) *
        gridDim.x;

    const float beta = 1.0f - alpha;

    for (
        int64_t i = thread_id;
        i < n;
        i += stride
    ) {
        const float va = to_float(a[i]);
        const float vb = to_float(b[i]);

        const float result =
            alpha * va +
            beta * vb;

        out[i] = from_float<scalar_t>(
            result
        );
    }
}


// ============================================================================
// CUDA KERNEL: FISHER MERGE
// ============================================================================

template <typename scalar_t>
__global__ void fisher_merge_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ fa,
    const scalar_t* __restrict__ fb,
    int64_t n
) {
    const int64_t thread_id =
        static_cast<int64_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const int64_t stride =
        static_cast<int64_t>(blockDim.x) *
        gridDim.x;

    for (
        int64_t i = thread_id;
        i < n;
        i += stride
    ) {
        const float va = to_float(a[i]);
        const float vb = to_float(b[i]);

        // Fisher values are expected to be non-negative.
        // Clamp pathological negative values defensively.
        const float raw_fa = to_float(fa[i]);
        const float raw_fb = to_float(fb[i]);

        const float f_a =
            fmaxf(raw_fa, 0.0f) + EPSILON;

        const float f_b =
            fmaxf(raw_fb, 0.0f) + EPSILON;

        const float denominator =
            f_a + f_b;

        const float result =
            (f_a * va + f_b * vb) /
            denominator;

        out[i] = from_float<scalar_t>(
            result
        );
    }
}


// ============================================================================
// CUDA KERNEL: SLERP
// ============================================================================
//
// The expensive geometric calculations are performed once by Python/C++
// before launching this elementwise kernel.
//
// out = sin((1-a)theta)/sin(theta) * A
//     + sin(a theta)/sin(theta) * B
//
// ============================================================================

template <typename scalar_t>
__global__ void slerp_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    float term_a,
    float term_b,
    int64_t n
) {
    const int64_t thread_id =
        static_cast<int64_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const int64_t stride =
        static_cast<int64_t>(blockDim.x) *
        gridDim.x;

    for (
        int64_t i = thread_id;
        i < n;
        i += stride
    ) {
        const float va = to_float(a[i]);
        const float vb = to_float(b[i]);

        const float result =
            term_a * va +
            term_b * vb;

        out[i] = from_float<scalar_t>(
            result
        );
    }
}


// ============================================================================
// CUDA KERNEL: TIES
// ============================================================================

template <typename scalar_t>
__global__ void ties_merge_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ mask_a,
    const scalar_t* __restrict__ mask_b,
    int64_t n
) {
    const int64_t thread_id =
        static_cast<int64_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const int64_t stride =
        static_cast<int64_t>(blockDim.x) *
        gridDim.x;

    for (
        int64_t i = thread_id;
        i < n;
        i += stride
    ) {
        const float va = to_float(a[i]);
        const float vb = to_float(b[i]);

        const float ma =
            to_float(mask_a[i]);

        const float mb =
            to_float(mask_b[i]);

        const int sign_a =
            (va > 0.0f) -
            (va < 0.0f);

        const int sign_b =
            (vb > 0.0f) -
            (vb < 0.0f);

        const bool compatible =
            sign_a != 0 &&
            sign_a == sign_b &&
            ma > 0.5f &&
            mb > 0.5f;

        const float result =
            compatible
                ? 0.5f * (va + vb)
                : va;

        out[i] = from_float<scalar_t>(
            result
        );
    }
}


// ============================================================================
// CUDA: WEIGHTED AVERAGE
// ============================================================================

torch::Tensor weighted_avg_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    float alpha
) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");

    validate_merge_pair(
        a,
        b,
        "weighted_avg_cuda"
    );

    validate_alpha(
        alpha,
        "weighted_avg_cuda"
    );

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    const int blocks =
        static_cast<int>(
            calculate_blocks(n)
        );

    const cudaStream_t stream =
        pytorch_current_stream(a);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_weighted_avg_cuda",
        [&] {
            weighted_avg_kernel<scalar_t>
                <<<blocks, DEFAULT_THREADS, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    alpha,
                    n
                );
        }
    );

    check_cuda_launch(
        "weighted_avg_cuda"
    );

    return out;
}


// ============================================================================
// CUDA: FISHER MERGE
// ============================================================================

torch::Tensor fisher_merge_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& fa,
    const torch::Tensor& fb
) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    check_cuda_tensor(fa, "fa");
    check_cuda_tensor(fb, "fb");

    validate_merge_pair(
        a,
        b,
        "fisher_merge_cuda"
    );

    TORCH_CHECK(
        fa.sizes() == a.sizes(),
        "fisher_merge_cuda: fa shape ",
        fa.sizes(),
        " does not match ",
        a.sizes()
    );

    TORCH_CHECK(
        fb.sizes() == a.sizes(),
        "fisher_merge_cuda: fb shape ",
        fb.sizes(),
        " does not match ",
        a.sizes()
    );

    check_same_device(
        a,
        fa,
        "fisher_merge_cuda"
    );

    check_same_device(
        a,
        fb,
        "fisher_merge_cuda"
    );

    check_same_dtype(
        a,
        fa,
        "fisher_merge_cuda"
    );

    check_same_dtype(
        a,
        fb,
        "fisher_merge_cuda"
    );

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    const int blocks =
        static_cast<int>(
            calculate_blocks(n)
        );

    const cudaStream_t stream =
        pytorch_current_stream(a);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_fisher_merge_cuda",
        [&] {
            fisher_merge_kernel<scalar_t>
                <<<blocks, DEFAULT_THREADS, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    fa.data_ptr<scalar_t>(),
                    fb.data_ptr<scalar_t>(),
                    n
                );
        }
    );

    check_cuda_launch(
        "fisher_merge_cuda"
    );

    return out;
}


// ============================================================================
// CUDA: SLERP
// ============================================================================

torch::Tensor slerp_merge_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    float alpha,
    float omega,
    float sin_omega
) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");

    validate_merge_pair(
        a,
        b,
        "slerp_merge_cuda"
    );

    validate_alpha(
        alpha,
        "slerp_merge_cuda"
    );

    TORCH_CHECK(
        std::isfinite(omega),
        "slerp_merge_cuda: omega must be finite"
    );

    TORCH_CHECK(
        std::isfinite(sin_omega),
        "slerp_merge_cuda: sin_omega must be finite"
    );

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    // Near-zero angular distance is numerically unstable for SLERP.
    // Fall back to linear interpolation.
    if (std::abs(sin_omega) < 1e-6f) {
        return weighted_avg_cuda(
            a,
            b,
            alpha
        );
    }

    const float inv_sin =
        1.0f / sin_omega;

    const float term_a =
        std::sin(
            (1.0f - alpha) * omega
        ) * inv_sin;

    const float term_b =
        std::sin(
            alpha * omega
        ) * inv_sin;

    const int blocks =
        static_cast<int>(
            calculate_blocks(n)
        );

    const cudaStream_t stream =
        pytorch_current_stream(a);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_slerp_cuda",
        [&] {
            slerp_kernel<scalar_t>
                <<<blocks, DEFAULT_THREADS, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    term_a,
                    term_b,
                    n
                );
        }
    );

    check_cuda_launch(
        "slerp_merge_cuda"
    );

    return out;
}


// ============================================================================
// CUDA: TIES
// ============================================================================

torch::Tensor ties_merge_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& mask_a,
    const torch::Tensor& mask_b
) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    check_cuda_tensor(mask_a, "mask_a");
    check_cuda_tensor(mask_b, "mask_b");

    validate_merge_pair(
        a,
        b,
        "ties_merge_cuda"
    );

    TORCH_CHECK(
        mask_a.sizes() == a.sizes(),
        "ties_merge_cuda: mask_a shape ",
        mask_a.sizes(),
        " does not match ",
        a.sizes()
    );

    TORCH_CHECK(
        mask_b.sizes() == a.sizes(),
        "ties_merge_cuda: mask_b shape ",
        mask_b.sizes(),
        " does not match ",
        a.sizes()
    );

    check_same_device(
        a,
        mask_a,
        "ties_merge_cuda"
    );

    check_same_device(
        a,
        mask_b,
        "ties_merge_cuda"
    );

    check_same_dtype(
        a,
        mask_a,
        "ties_merge_cuda"
    );

    check_same_dtype(
        a,
        mask_b,
        "ties_merge_cuda"
    );

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    const int blocks =
        static_cast<int>(
            calculate_blocks(n)
        );

    const cudaStream_t stream =
        pytorch_current_stream(a);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_ties_cuda",
        [&] {
            ties_merge_kernel<scalar_t>
                <<<blocks, DEFAULT_THREADS, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    a.data_ptr<scalar_t>(),
                    b.data_ptr<scalar_t>(),
                    mask_a.data_ptr<scalar_t>(),
                    mask_b.data_ptr<scalar_t>(),
                    n
                );
        }
    );

    check_cuda_launch(
        "ties_merge_cuda"
    );

    return out;
}


// ============================================================================
// CPU: WEIGHTED AVERAGE
// ============================================================================

torch::Tensor weighted_avg_cpu(
    const torch::Tensor& a,
    const torch::Tensor& b,
    float alpha
) {
    check_cpu_tensor(a, "a");
    check_cpu_tensor(b, "b");

    validate_merge_pair(
        a,
        b,
        "weighted_avg_cpu"
    );

    validate_alpha(
        alpha,
        "weighted_avg_cpu"
    );

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_weighted_avg_cpu",
        [&] {
            const scalar_t* ap =
                a.data_ptr<scalar_t>();

            const scalar_t* bp =
                b.data_ptr<scalar_t>();

            scalar_t* op =
                out.data_ptr<scalar_t>();

            const float beta =
                1.0f - alpha;

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                const float va =
                    static_cast<float>(ap[i]);

                const float vb =
                    static_cast<float>(bp[i]);

                op[i] =
                    static_cast<scalar_t>(
                        alpha * va +
                        beta * vb
                    );
            }
        }
    );

    return out;
}


// ============================================================================
// CPU: FISHER MERGE
// ============================================================================

torch::Tensor fisher_merge_cpu(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& fa,
    const torch::Tensor& fb
) {
    check_cpu_tensor(a, "a");
    check_cpu_tensor(b, "b");
    check_cpu_tensor(fa, "fa");
    check_cpu_tensor(fb, "fb");

    validate_merge_pair(
        a,
        b,
        "fisher_merge_cpu"
    );

    TORCH_CHECK(
        fa.sizes() == a.sizes(),
        "fisher_merge_cpu: fa shape mismatch"
    );

    TORCH_CHECK(
        fb.sizes() == a.sizes(),
        "fisher_merge_cpu: fb shape mismatch"
    );

    check_same_dtype(
        a,
        fa,
        "fisher_merge_cpu"
    );

    check_same_dtype(
        a,
        fb,
        "fisher_merge_cpu"
    );

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_fisher_merge_cpu",
        [&] {
            const scalar_t* ap =
                a.data_ptr<scalar_t>();

            const scalar_t* bp =
                b.data_ptr<scalar_t>();

            const scalar_t* fap =
                fa.data_ptr<scalar_t>();

            const scalar_t* fbp =
                fb.data_ptr<scalar_t>();

            scalar_t* op =
                out.data_ptr<scalar_t>();

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                const float va =
                    static_cast<float>(ap[i]);

                const float vb =
                    static_cast<float>(bp[i]);

                const float raw_fa =
                    static_cast<float>(fap[i]);

                const float raw_fb =
                    static_cast<float>(fbp[i]);

                const float f_a =
                    std::max(raw_fa, 0.0f) +
                    EPSILON;

                const float f_b =
                    std::max(raw_fb, 0.0f) +
                    EPSILON;

                op[i] =
                    static_cast<scalar_t>(
                        (f_a * va +
                         f_b * vb) /
                        (f_a + f_b)
                    );
            }
        }
    );

    return out;
}


// ============================================================================
// CPU: SLERP
// ============================================================================

torch::Tensor slerp_merge_cpu(
    const torch::Tensor& a,
    const torch::Tensor& b,
    float alpha,
    float omega,
    float sin_omega
) {
    check_cpu_tensor(a, "a");
    check_cpu_tensor(b, "b");

    validate_merge_pair(
        a,
        b,
        "slerp_merge_cpu"
    );

    validate_alpha(
        alpha,
        "slerp_merge_cpu"
    );

    TORCH_CHECK(
        std::isfinite(omega),
        "slerp_merge_cpu: omega must be finite"
    );

    TORCH_CHECK(
        std::isfinite(sin_omega),
        "slerp_merge_cpu: sin_omega must be finite"
    );

    if (std::abs(sin_omega) < 1e-6f) {
        return weighted_avg_cpu(
            a,
            b,
            alpha
        );
    }

    const float inv_sin =
        1.0f / sin_omega;

    const float term_a =
        std::sin(
            (1.0f - alpha) * omega
        ) * inv_sin;

    const float term_b =
        std::sin(
            alpha * omega
        ) * inv_sin;

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_slerp_cpu",
        [&] {
            const scalar_t* ap =
                a.data_ptr<scalar_t>();

            const scalar_t* bp =
                b.data_ptr<scalar_t>();

            scalar_t* op =
                out.data_ptr<scalar_t>();

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                const float va =
                    static_cast<float>(ap[i]);

                const float vb =
                    static_cast<float>(bp[i]);

                op[i] =
                    static_cast<scalar_t>(
                        term_a * va +
                        term_b * vb
                    );
            }
        }
    );

    return out;
}


// ============================================================================
// CPU: TIES
// ============================================================================

torch::Tensor ties_merge_cpu(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& mask_a,
    const torch::Tensor& mask_b
) {
    check_cpu_tensor(a, "a");
    check_cpu_tensor(b, "b");
    check_cpu_tensor(mask_a, "mask_a");
    check_cpu_tensor(mask_b, "mask_b");

    validate_merge_pair(
        a,
        b,
        "ties_merge_cpu"
    );

    TORCH_CHECK(
        mask_a.sizes() == a.sizes(),
        "ties_merge_cpu: mask_a shape mismatch"
    );

    TORCH_CHECK(
        mask_b.sizes() == a.sizes(),
        "ties_merge_cpu: mask_b shape mismatch"
    );

    check_same_dtype(
        a,
        mask_a,
        "ties_merge_cpu"
    );

    check_same_dtype(
        a,
        mask_b,
        "ties_merge_cpu"
    );

    auto out = torch::empty_like(a);

    const int64_t n = a.numel();

    if (n == 0) {
        return out;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        a.scalar_type(),
        "ftrain_ties_cpu",
        [&] {
            const scalar_t* ap =
                a.data_ptr<scalar_t>();

            const scalar_t* bp =
                b.data_ptr<scalar_t>();

            const scalar_t* map =
                mask_a.data_ptr<scalar_t>();

            const scalar_t* mbp =
                mask_b.data_ptr<scalar_t>();

            scalar_t* op =
                out.data_ptr<scalar_t>();

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                const float va =
                    static_cast<float>(ap[i]);

                const float vb =
                    static_cast<float>(bp[i]);

                const float ma =
                    static_cast<float>(map[i]);

                const float mb =
                    static_cast<float>(mbp[i]);

                const int sign_a =
                    (va > 0.0f) -
                    (va < 0.0f);

                const int sign_b =
                    (vb > 0.0f) -
                    (vb < 0.0f);

                const bool compatible =
                    sign_a != 0 &&
                    sign_a == sign_b &&
                    ma > 0.5f &&
                    mb > 0.5f;

                const float result =
                    compatible
                        ? 0.5f * (va + vb)
                        : va;

                op[i] =
                    static_cast<scalar_t>(
                        result
                    );
            }
        }
    );

    return out;
}


// ============================================================================
// PYBIND11
// ============================================================================

} // namespace ftrain


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.doc() =
        "FTRAIN high-performance model merge kernels";

    // ------------------------------------------------------------------------
    // Weighted average
    // ------------------------------------------------------------------------

    m.def(
        "weighted_avg_cuda",
        &ftrain::weighted_avg_cuda,
        "FTRAIN CUDA weighted average merge"
    );

    m.def(
        "weighted_avg_cpu",
        &ftrain::weighted_avg_cpu,
        "FTRAIN CPU/OpenMP weighted average merge"
    );

    // ------------------------------------------------------------------------
    // Fisher
    // ------------------------------------------------------------------------

    m.def(
        "fisher_merge_cuda",
        &ftrain::fisher_merge_cuda,
        "FTRAIN CUDA Fisher-weighted merge"
    );

    m.def(
        "fisher_merge_cpu",
        &ftrain::fisher_merge_cpu,
        "FTRAIN CPU/OpenMP Fisher-weighted merge"
    );

    // ------------------------------------------------------------------------
    // SLERP
    // ------------------------------------------------------------------------

    m.def(
        "slerp_merge_cuda",
        &ftrain::slerp_merge_cuda,
        "FTRAIN CUDA SLERP merge"
    );

    m.def(
        "slerp_merge_cpu",
        &ftrain::slerp_merge_cpu,
        "FTRAIN CPU/OpenMP SLERP merge"
    );

    // ------------------------------------------------------------------------
    // TIES
    // ------------------------------------------------------------------------

    m.def(
        "ties_merge_cuda",
        &ftrain::ties_merge_cuda,
        "FTRAIN CUDA TIES merge"
    );

    m.def(
        "ties_merge_cpu",
        &ftrain::ties_merge_cpu,
        "FTRAIN CPU/OpenMP TIES merge"
    );
}
