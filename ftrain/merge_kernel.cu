#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void weighted_avg_kernel(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = alpha * a[idx] + (1.0f - alpha) * b[idx];
}
torch::Tensor weighted_avg_cuda(const torch::Tensor a, const torch::Tensor b, float alpha) {
    TORCH_CHECK(a.sizes() == b.sizes()); TORCH_CHECK(a.device().is_cuda());
    auto out = torch::empty_like(a); int n = a.numel();
    int threads = 256, blocks = (n+threads-1)/threads;
    weighted_avg_kernel<<<blocks,threads>>>(out.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),alpha,n);
    return out;
}

__global__ void fisher_avg_kernel(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ fa, const float* __restrict__ fb, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float f_a = fa[idx] + 1e-8f, f_b = fb[idx] + 1e-8f;
        out[idx] = (f_a * a[idx] + f_b * b[idx]) / (f_a + f_b);
    }
}
torch::Tensor fisher_merge_cuda(const torch::Tensor a, const torch::Tensor b, const torch::Tensor fa, const torch::Tensor fb) {
    TORCH_CHECK(a.sizes()==b.sizes() && a.sizes()==fa.sizes() && a.sizes()==fb.sizes()); TORCH_CHECK(a.device().is_cuda());
    auto out = torch::empty_like(a); int n = a.numel(); int threads = 256, blocks = (n+threads-1)/threads;
    fisher_avg_kernel<<<blocks,threads>>>(out.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),fa.data_ptr<float>(),fb.data_ptr<float>(),n);
    return out;
}

__global__ void slerp_kernel(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, float omega, float sin_omega, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float term_a = __sinf((1.0f-alpha)*omega)/sin_omega, term_b = __sinf(alpha*omega)/sin_omega;
        out[idx] = term_a * a[idx] + term_b * b[idx];
    }
}
torch::Tensor slerp_merge_cuda(const torch::Tensor a, const torch::Tensor b, float alpha, float omega, float sin_omega) {
    TORCH_CHECK(a.sizes()==b.sizes() && a.device().is_cuda());
    auto out = torch::empty_like(a); int n = a.numel(); int threads = 256, blocks = (n+threads-1)/threads;
    slerp_kernel<<<blocks,threads>>>(out.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),omega,sin_omega,alpha,n);
    return out;
}

__global__ void ties_sparsify_kernel(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ mask_a, const float* __restrict__ mask_b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sign_a = (a[idx]>0?1.0f:(a[idx]<0?-1.0f:0.0f)), sign_b = (b[idx]>0?1.0f:(b[idx]<0?-1.0f:0.0f));
        float ma = mask_a[idx], mb = mask_b[idx];
        if (sign_a==sign_b && sign_a!=0.0f && ma>0.5f && mb>0.5f) out[idx] = (a[idx]+b[idx])*0.5f;
        else out[idx] = a[idx];
    }
}
torch::Tensor ties_merge_cuda(const torch::Tensor a, const torch::Tensor b, const torch::Tensor mask_a, const torch::Tensor mask_b) {
    TORCH_CHECK(a.sizes()==b.sizes() && a.sizes()==mask_a.sizes() && a.sizes()==mask_b.sizes()); TORCH_CHECK(a.device().is_cuda());
    auto out = torch::empty_like(a); int n = a.numel(); int threads = 256, blocks = (n+threads-1)/threads;
    ties_sparsify_kernel<<<blocks,threads>>>(out.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),mask_a.data_ptr<float>(),mask_b.data_ptr<float>(),n);
    return out;
}

torch::Tensor weighted_avg_cpu(const torch::Tensor a, const torch::Tensor b, float alpha) {
    TORCH_CHECK(a.sizes()==b.sizes()); auto out = torch::empty_like(a); auto ap=a.data_ptr<float>(), bp=b.data_ptr<float>(), op=out.data_ptr<float>(); int n = a.numel();
    #pragma omp parallel for simd
    for (int i=0;i<n;++i) op[i] = alpha*ap[i]+(1.0f-alpha)*bp[i];
    return out;
}
torch::Tensor fisher_merge_cpu(const torch::Tensor a, const torch::Tensor b, const torch::Tensor fa, const torch::Tensor fb) {
    TORCH_CHECK(a.sizes()==b.sizes() && a.sizes()==fa.sizes() && a.sizes()==fb.sizes()); auto out = torch::empty_like(a); auto ap=a.data_ptr<float>(), bp=b.data_ptr<float>(), fap=fa.data_ptr<float>(), fbp=fb.data_ptr<float>(), op=out.data_ptr<float>(); int n = a.numel();
    #pragma omp parallel for simd
    for (int i=0;i<n;++i) {
        float f_a = fap[i]+1e-8f, f_b = fbp[i]+1e-8f; op[i] = (f_a*ap[i] + f_b*bp[i]) / (f_a+f_b);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_avg_cuda", &weighted_avg_cuda); m.def("weighted_avg_cpu", &weighted_avg_cpu);
    m.def("fisher_merge_cuda", &fisher_merge_cuda); m.def("fisher_merge_cpu", &fisher_merge_cpu);
    m.def("slerp_merge_cuda", &slerp_merge_cuda); m.def("ties_merge_cuda", &ties_merge_cuda);
}
