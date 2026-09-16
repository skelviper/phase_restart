#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kEpsilon = 1.0e-6;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

// Field layout for the full ordered-row reduction.
enum FullField : int {
    kFullRCis = 0,
    kFullRInter = 1,
    kFullDP = 2,
    kFullRepCross = 3,
    kFullRepHomolog = 4,
    kFullGradCisX0 = 5,
    kFullGradCisX1 = 8,
    kFullGradInterX0 = 11,
    kFullGradInterX1 = 14,
    kFullGradRepX0 = 17,
    kFullGradRepX1 = 20,
    kFullFields = 23,
};

// Field layout for the sparse observed-row reduction.
enum SparseField : int {
    kSparseLogCis = 0,
    kSparseLogInter = 1,
    kSparseDP = 2,
    kSparseGradX0 = 3,
    kSparseGradX1 = 6,
    kSparseFields = 9,
};

struct KernelTerm {
    double value;
    double grad[3];
};

struct ContactTerms {
    double rate;
    double drdp;
    double grad_x0[3];
    double grad_x1[3];
};

__device__ __forceinline__ double coordinate(const double* x, int copy, int locus,
                                             int dimension, int n_loci) {
    const int64_t index = (static_cast<int64_t>(copy) * n_loci + locus) * 3 + dimension;
    return x[index];
}

__device__ __forceinline__ void difference(const double* x, int n_loci,
                                           int copy_i, int i, int copy_j, int j,
                                           double out[3]) {
    out[0] = coordinate(x, copy_i, i, 0, n_loci) - coordinate(x, copy_j, j, 0, n_loci);
    out[1] = coordinate(x, copy_i, i, 1, n_loci) - coordinate(x, copy_j, j, 1, n_loci);
    out[2] = coordinate(x, copy_i, i, 2, n_loci) - coordinate(x, copy_j, j, 2, n_loci);
}

__device__ __forceinline__ KernelTerm contact_kernel(const double delta[3],
                                                     double inv_r0_sq) {
    const double norm_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    const double base = 1.0 + norm_sq * inv_r0_sq;
    const double inv_base = 1.0 / base;
    const double inv_base_sq = inv_base * inv_base;
    const double inv_base_cu = inv_base_sq * inv_base;
    KernelTerm result;
    result.value = kEpsilon + (1.0 - kEpsilon) * inv_base_sq;
    const double coefficient = -4.0 * (1.0 - kEpsilon) * inv_r0_sq * inv_base_cu;
    result.grad[0] = coefficient * delta[0];
    result.grad[1] = coefficient * delta[1];
    result.grad[2] = coefficient * delta[2];
    return result;
}

__device__ __forceinline__ KernelTerm repulsion_kernel(const double delta[3],
                                                       double threshold) {
    const double norm_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    const double distance = sqrt(norm_sq);
    const double raw_hinge = 1.0 - distance / threshold;
    KernelTerm result;
    if (raw_hinge > 0.0) {
        result.value = raw_hinge * raw_hinge;
        // The archived NumPy source chooses the zero vector at distance zero.
        const double coefficient = distance > 0.0
            ? (-2.0 * raw_hinge / threshold / distance)
            : 0.0;
        result.grad[0] = coefficient * delta[0];
        result.grad[1] = coefficient * delta[1];
        result.grad[2] = coefficient * delta[2];
    } else {
        result.value = 0.0;
        result.grad[0] = 0.0;
        result.grad[1] = 0.0;
        result.grad[2] = 0.0;
    }
    return result;
}

__device__ __forceinline__ ContactTerms make_contact_terms(
    const double* x, const double* exposure, const int32_t* chromosome,
    int n_loci, int i, int j, double p, double inv_r0_sq) {
    double aa[3], ab[3], ba[3], bb[3];
    difference(x, n_loci, 0, i, 0, j, aa);
    difference(x, n_loci, 0, i, 1, j, ab);
    difference(x, n_loci, 1, i, 0, j, ba);
    difference(x, n_loci, 1, i, 1, j, bb);
    const KernelTerm kaa = contact_kernel(aa, inv_r0_sq);
    const KernelTerm kab = contact_kernel(ab, inv_r0_sq);
    const KernelTerm kba = contact_kernel(ba, inv_r0_sq);
    const KernelTerm kbb = contact_kernel(bb, inv_r0_sq);
    const bool cis = chromosome[i] == chromosome[j];
    const double same_coefficient = cis ? 0.5 * p : 0.25;
    const double cross_coefficient = cis ? 0.5 * (1.0 - p) : 0.25;
    const double same = kaa.value + kbb.value;
    const double cross = kab.value + kba.value;
    const double exposure_product = exposure[i] * exposure[j];

    ContactTerms result;
    result.rate = exposure_product * (same_coefficient * same + cross_coefficient * cross);
    result.drdp = cis ? exposure_product * 0.5 * (same - cross) : 0.0;
    for (int dimension = 0; dimension < 3; ++dimension) {
        result.grad_x0[dimension] = exposure_product * (
            same_coefficient * kaa.grad[dimension] + cross_coefficient * kab.grad[dimension]);
        result.grad_x1[dimension] = exposure_product * (
            cross_coefficient * kba.grad[dimension] + same_coefficient * kbb.grad[dimension]);
    }
    return result;
}

// Each block owns one center locus. The loop over all other loci is ordered and
// duplicated in the opposite center row, so every endpoint gradient is local
// to exactly one block and needs no global coordinate atomic.
__device__ __forceinline__ void reduce_fields(const double* values, double* reduced,
                                              int field_count) {
    __shared__ double warp_sums[kFullFields * kWarps];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    for (int field = 0; field < field_count; ++field) {
        double value = values[field];
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            warp_sums[field * kWarps + warp] = value;
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int field = 0; field < field_count; ++field) {
            double value = lane < kWarps ? warp_sums[field * kWarps + lane] : 0.0;
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffu, value, offset);
            }
            if (lane == 0) {
                reduced[field] = value;
            }
        }
    }
}

__global__ void full_ordered_rows_kernel(
    const double* x, const double* exposure, const int32_t* chromosome, int n_loci,
    double p, double inv_r0_sq, double repulsion_threshold,
    double* row_r_cis, double* row_r_inter, double* row_dp,
    double* row_rep_cross, double* row_rep_homolog,
    double* grad_r_cis, double* grad_r_inter, double* grad_rep) {
    const int i = static_cast<int>(blockIdx.x);
    if (i >= n_loci) {
        return;
    }
    double values[kFullFields] = {};
    for (int j = threadIdx.x; j < n_loci; j += blockDim.x) {
        if (j == i) {
            continue;
        }
        const ContactTerms contact = make_contact_terms(
            x, exposure, chromosome, n_loci, i, j, p, inv_r0_sq);
        if (chromosome[i] == chromosome[j]) {
            values[kFullRCis] += contact.rate;
            values[kFullDP] += contact.drdp;
            for (int dimension = 0; dimension < 3; ++dimension) {
                values[kFullGradCisX0 + dimension] += contact.grad_x0[dimension];
                values[kFullGradCisX1 + dimension] += contact.grad_x1[dimension];
            }
        } else {
            values[kFullRInter] += contact.rate;
            for (int dimension = 0; dimension < 3; ++dimension) {
                values[kFullGradInterX0 + dimension] += contact.grad_x0[dimension];
                values[kFullGradInterX1 + dimension] += contact.grad_x1[dimension];
            }
        }

        double delta[3];
        difference(x, n_loci, 0, i, 0, j, delta);
        const KernelTerm rep_aa = repulsion_kernel(delta, repulsion_threshold);
        difference(x, n_loci, 0, i, 1, j, delta);
        const KernelTerm rep_ab = repulsion_kernel(delta, repulsion_threshold);
        difference(x, n_loci, 1, i, 0, j, delta);
        const KernelTerm rep_ba = repulsion_kernel(delta, repulsion_threshold);
        difference(x, n_loci, 1, i, 1, j, delta);
        const KernelTerm rep_bb = repulsion_kernel(delta, repulsion_threshold);
        values[kFullRepCross] += rep_aa.value + rep_ab.value + rep_ba.value + rep_bb.value;
        for (int dimension = 0; dimension < 3; ++dimension) {
            values[kFullGradRepX0 + dimension] += rep_aa.grad[dimension] + rep_ab.grad[dimension];
            values[kFullGradRepX1 + dimension] += rep_ba.grad[dimension] + rep_bb.grad[dimension];
        }
    }

    if (threadIdx.x == 0) {
        double delta[3];
        difference(x, n_loci, 0, i, 1, i, delta);
        const KernelTerm homolog = repulsion_kernel(delta, repulsion_threshold);
        values[kFullRepHomolog] = homolog.value;
        for (int dimension = 0; dimension < 3; ++dimension) {
            values[kFullGradRepX0 + dimension] += homolog.grad[dimension];
            values[kFullGradRepX1 + dimension] -= homolog.grad[dimension];
        }
    }

    double reduced[kFullFields] = {};
    reduce_fields(values, reduced, kFullFields);
    if (threadIdx.x != 0) {
        return;
    }
    row_r_cis[i] = reduced[kFullRCis];
    row_r_inter[i] = reduced[kFullRInter];
    row_dp[i] = reduced[kFullDP];
    row_rep_cross[i] = reduced[kFullRepCross];
    row_rep_homolog[i] = reduced[kFullRepHomolog];
    for (int dimension = 0; dimension < 3; ++dimension) {
        grad_r_cis[(0 * n_loci + i) * 3 + dimension] = reduced[kFullGradCisX0 + dimension];
        grad_r_cis[(1 * n_loci + i) * 3 + dimension] = reduced[kFullGradCisX1 + dimension];
        grad_r_inter[(0 * n_loci + i) * 3 + dimension] = reduced[kFullGradInterX0 + dimension];
        grad_r_inter[(1 * n_loci + i) * 3 + dimension] = reduced[kFullGradInterX1 + dimension];
        grad_rep[(0 * n_loci + i) * 3 + dimension] = reduced[kFullGradRepX0 + dimension];
        grad_rep[(1 * n_loci + i) * 3 + dimension] = reduced[kFullGradRepX1 + dimension];
    }
}

__global__ void reduce_full_rows_kernel(
    const double* row_r_cis, const double* row_r_inter, const double* row_dp,
    const double* row_rep_cross, const double* row_rep_homolog, int n_loci,
    double* summary) {
    double values[5] = {};
    for (int index = threadIdx.x; index < n_loci; index += blockDim.x) {
        values[0] += row_r_cis[index];
        values[1] += row_r_inter[index];
        values[2] += row_dp[index];
        values[3] += row_rep_cross[index];
        values[4] += row_rep_homolog[index];
    }
    double reduced[5] = {};
    reduce_fields(values, reduced, 5);
    if (threadIdx.x == 0) {
        for (int field = 0; field < 5; ++field) {
            summary[field] = reduced[field];
        }
    }
}

__global__ void sparse_observed_rows_kernel(
    const double* x, const double* exposure, const int32_t* chromosome, int n_loci,
    const int64_t* out_ptr, const int32_t* out_j, const int64_t* out_count,
    const int64_t* in_ptr, const int32_t* in_i, const int64_t* in_count,
    double p, double inv_r0_sq,
    double* row_log_cis, double* row_log_inter, double* row_dp, double* grad_observed) {
    const int center = static_cast<int>(blockIdx.x);
    if (center >= n_loci) {
        return;
    }
    double values[kSparseFields] = {};
    for (int64_t edge = out_ptr[center] + threadIdx.x;
         edge < out_ptr[center + 1]; edge += blockDim.x) {
        const int partner = out_j[edge];
        const int64_t integer_count = out_count[edge];
        if (integer_count <= 0) {
            continue;
        }
        const ContactTerms contact = make_contact_terms(
            x, exposure, chromosome, n_loci, center, partner, p, inv_r0_sq);
        const double count = static_cast<double>(integer_count);
        const double log_term = count * log(contact.rate);
        const double log_gradient_weight = count / contact.rate;
        if (chromosome[center] == chromosome[partner]) {
            values[kSparseLogCis] += log_term;
        } else {
            values[kSparseLogInter] += log_term;
        }
        values[kSparseDP] += log_gradient_weight * contact.drdp;
        for (int dimension = 0; dimension < 3; ++dimension) {
            values[kSparseGradX0 + dimension] += log_gradient_weight * contact.grad_x0[dimension];
            values[kSparseGradX1 + dimension] += log_gradient_weight * contact.grad_x1[dimension];
        }
    }
    for (int64_t edge = in_ptr[center] + threadIdx.x;
         edge < in_ptr[center + 1]; edge += blockDim.x) {
        const int partner = in_i[edge];
        const int64_t integer_count = in_count[edge];
        if (integer_count <= 0) {
            continue;
        }
        // This is the same upper-triangle edge in reverse orientation. Only
        // the central endpoint gradient is added; log/p are counted outwards.
        const ContactTerms contact = make_contact_terms(
            x, exposure, chromosome, n_loci, center, partner, p, inv_r0_sq);
        const double log_gradient_weight = static_cast<double>(integer_count) / contact.rate;
        for (int dimension = 0; dimension < 3; ++dimension) {
            values[kSparseGradX0 + dimension] += log_gradient_weight * contact.grad_x0[dimension];
            values[kSparseGradX1 + dimension] += log_gradient_weight * contact.grad_x1[dimension];
        }
    }

    double reduced[kSparseFields] = {};
    reduce_fields(values, reduced, kSparseFields);
    if (threadIdx.x != 0) {
        return;
    }
    row_log_cis[center] = reduced[kSparseLogCis];
    row_log_inter[center] = reduced[kSparseLogInter];
    row_dp[center] = reduced[kSparseDP];
    for (int dimension = 0; dimension < 3; ++dimension) {
        grad_observed[(0 * n_loci + center) * 3 + dimension] = reduced[kSparseGradX0 + dimension];
        grad_observed[(1 * n_loci + center) * 3 + dimension] = reduced[kSparseGradX1 + dimension];
    }
}

__global__ void reduce_sparse_rows_kernel(
    const double* row_log_cis, const double* row_log_inter, const double* row_dp,
    int n_loci, double* summary) {
    double values[3] = {};
    for (int index = threadIdx.x; index < n_loci; index += blockDim.x) {
        values[0] += row_log_cis[index];
        values[1] += row_log_inter[index];
        values[2] += row_dp[index];
    }
    double reduced[3] = {};
    reduce_fields(values, reduced, 3);
    if (threadIdx.x == 0) {
        for (int field = 0; field < 3; ++field) {
            summary[field] = reduced[field];
        }
    }
}

void check_cuda_double(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kDouble, name, " must be float64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int32(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int64(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const at::Tensor& tensor, const char* name, int device_index) {
    TORCH_CHECK(tensor.get_device() == device_index, name, " is on a different CUDA device");
}

std::vector<at::Tensor> full_terms(
    const at::Tensor& x, const at::Tensor& exposure, const at::Tensor& chromosome,
    double p, double r0, double repulsion_threshold) {
    check_cuda_double(x, "x");
    const int device_index = x.get_device();
    check_cuda_double(exposure, "exposure");
    check_cuda_int32(chromosome, "chromosome");
    check_same_device(exposure, "exposure", device_index);
    check_same_device(chromosome, "chromosome", device_index);
    c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.dim() == 3 && x.size(0) == 2 && x.size(2) == 3,
                "x must have shape (2, n_loci, 3)");
    const int64_t n_loci_64 = x.size(1);
    TORCH_CHECK(n_loci_64 > 0 && n_loci_64 <= std::numeric_limits<int>::max(),
                "n_loci must fit a positive int32");
    const int n_loci = static_cast<int>(n_loci_64);
    TORCH_CHECK(exposure.dim() == 1 && exposure.size(0) == n_loci_64,
                "exposure shape mismatch");
    TORCH_CHECK(chromosome.dim() == 1 && chromosome.size(0) == n_loci_64,
                "chromosome shape mismatch");
    TORCH_CHECK(std::isfinite(p) && std::isfinite(r0) && r0 > 0.0 &&
                    std::isfinite(repulsion_threshold) && repulsion_threshold > 0.0,
                "nonfinite or nonpositive kernel parameter");

    auto options = x.options().dtype(at::kDouble);
    auto row_r_cis = at::empty({n_loci}, options);
    auto row_r_inter = at::empty({n_loci}, options);
    auto row_dp = at::empty({n_loci}, options);
    auto row_rep_cross = at::empty({n_loci}, options);
    auto row_rep_homolog = at::empty({n_loci}, options);
    auto grad_r_cis = at::empty({2, n_loci, 3}, options);
    auto grad_r_inter = at::empty({2, n_loci, 3}, options);
    auto grad_rep = at::empty({2, n_loci, 3}, options);
    auto summary = at::empty({5}, options);

    const double inv_r0_sq = 1.0 / (r0 * r0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    full_ordered_rows_kernel<<<n_loci, kThreads, 0, stream>>>(
        x.data_ptr<double>(), exposure.data_ptr<double>(), chromosome.data_ptr<int32_t>(),
        n_loci, p, inv_r0_sq, repulsion_threshold,
        row_r_cis.data_ptr<double>(), row_r_inter.data_ptr<double>(), row_dp.data_ptr<double>(),
        row_rep_cross.data_ptr<double>(), row_rep_homolog.data_ptr<double>(),
        grad_r_cis.data_ptr<double>(), grad_r_inter.data_ptr<double>(), grad_rep.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_full_rows_kernel<<<1, kThreads, 0, stream>>>(
        row_r_cis.data_ptr<double>(), row_r_inter.data_ptr<double>(), row_dp.data_ptr<double>(),
        row_rep_cross.data_ptr<double>(), row_rep_homolog.data_ptr<double>(), n_loci,
        summary.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {summary, grad_r_cis, grad_r_inter, grad_rep};
}

std::vector<at::Tensor> sparse_terms(
    const at::Tensor& x, const at::Tensor& exposure, const at::Tensor& chromosome,
    const at::Tensor& out_ptr, const at::Tensor& out_j, const at::Tensor& out_count,
    const at::Tensor& in_ptr, const at::Tensor& in_i, const at::Tensor& in_count,
    double p, double r0) {
    check_cuda_double(x, "x");
    const int device_index = x.get_device();
    check_cuda_double(exposure, "exposure");
    check_cuda_int32(chromosome, "chromosome");
    check_same_device(exposure, "exposure", device_index);
    check_same_device(chromosome, "chromosome", device_index);
    c10::cuda::CUDAGuard device_guard(x.device());
    check_cuda_int64(out_ptr, "out_ptr");
    check_cuda_int32(out_j, "out_j");
    check_cuda_int64(out_count, "out_count");
    check_cuda_int64(in_ptr, "in_ptr");
    check_cuda_int32(in_i, "in_i");
    check_cuda_int64(in_count, "in_count");
    check_same_device(out_ptr, "out_ptr", device_index);
    check_same_device(out_j, "out_j", device_index);
    check_same_device(out_count, "out_count", device_index);
    check_same_device(in_ptr, "in_ptr", device_index);
    check_same_device(in_i, "in_i", device_index);
    check_same_device(in_count, "in_count", device_index);
    TORCH_CHECK(x.dim() == 3 && x.size(0) == 2 && x.size(2) == 3,
                "x must have shape (2, n_loci, 3)");
    const int64_t n_loci_64 = x.size(1);
    TORCH_CHECK(n_loci_64 > 0 && n_loci_64 <= std::numeric_limits<int>::max(),
                "n_loci must fit a positive int32");
    const int n_loci = static_cast<int>(n_loci_64);
    TORCH_CHECK(exposure.dim() == 1 && exposure.size(0) == n_loci_64,
                "exposure shape mismatch");
    TORCH_CHECK(chromosome.dim() == 1 && chromosome.size(0) == n_loci_64,
                "chromosome shape mismatch");
    TORCH_CHECK(out_ptr.dim() == 1 && in_ptr.dim() == 1 &&
                    out_ptr.size(0) == n_loci_64 + 1 && in_ptr.size(0) == n_loci_64 + 1,
                "CSR pointer shape mismatch");
    TORCH_CHECK(out_j.dim() == 1 && out_count.dim() == 1 &&
                    in_i.dim() == 1 && in_count.dim() == 1 &&
                    out_j.size(0) == out_count.size(0) &&
                    in_i.size(0) == in_count.size(0),
                "CSR edge shape mismatch");
    TORCH_CHECK(std::isfinite(p) && std::isfinite(r0) && r0 > 0.0,
                "nonfinite or nonpositive kernel parameter");

    auto options = x.options().dtype(at::kDouble);
    auto row_log_cis = at::empty({n_loci}, options);
    auto row_log_inter = at::empty({n_loci}, options);
    auto row_dp = at::empty({n_loci}, options);
    auto grad_observed = at::empty({2, n_loci, 3}, options);
    auto summary = at::empty({3}, options);

    const double inv_r0_sq = 1.0 / (r0 * r0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    sparse_observed_rows_kernel<<<n_loci, kThreads, 0, stream>>>(
        x.data_ptr<double>(), exposure.data_ptr<double>(), chromosome.data_ptr<int32_t>(), n_loci,
        out_ptr.data_ptr<int64_t>(), out_j.data_ptr<int32_t>(), out_count.data_ptr<int64_t>(),
        in_ptr.data_ptr<int64_t>(), in_i.data_ptr<int32_t>(), in_count.data_ptr<int64_t>(),
        p, inv_r0_sq,
        row_log_cis.data_ptr<double>(), row_log_inter.data_ptr<double>(), row_dp.data_ptr<double>(),
        grad_observed.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_sparse_rows_kernel<<<1, kThreads, 0, stream>>>(
        row_log_cis.data_ptr<double>(), row_log_inter.data_ptr<double>(), row_dp.data_ptr<double>(),
        n_loci, summary.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {summary, grad_observed};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("full_terms", &full_terms, "Fused ordered full-grid contact and repulsion terms");
    module.def("sparse_terms", &sparse_terms, "Fused sparse observed contact terms");
}
