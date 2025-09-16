
// Key changes vs the baseline:
// 1) Shared-memory kernels (first step + leapfrog) that operate on interior points [1, N-2],
//    loading (blockDim.x + 2) values (with 1-point halos) to eliminate warp divergence
//    and reduce global memory traffic.
// 2) Single set of CUDA events/stream reused across all problem sizes; device buffers
//    sized for max N and reused to avoid per-case cudaMalloc/Free and EventCreate/Destroy.
// 3) Pinned host memory + cudaMemcpyAsync + single stream for simple overlap and lower H2D/D2H latency.
// 4) Optional GPU-side error reduction (per-block partials, no atomics) to avoid O(N) CPU reductions.
// 5) Boundary values remain 0 without branch writes inside kernels.
// 6) Clean timing: H2D, Kernel (solver only), D2H, Total.


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

// ============================= Config & Macros ===============================
#define ENABLE_KERNEL_CHECKS 0     // set 1 for debug (adds cudaPeekAtLastError per launch)
#define USE_PINNED_HOST      1     // host allocations with cudaHostAlloc
#define USE_GPU_ERROR        1     // compute L1/L2/relL2 on GPU via partial reductions

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr double V       = 1.0;     
static constexpr double CFL     = 0.50;    
static constexpr double Tfinal  = 10.23;   

// Problem sizes
static const int Ns[]   = {4096, 8192, 16384, 32768};
static const int nCases = int(sizeof(Ns) / sizeof(Ns[0]));

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s (%d) at %s:%d\n", \
                cudaGetErrorString(_e), _e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================ Exact Solution =============================
__host__ __device__ inline double exact(double x, double t) {
    return sin(M_PI * x) * cos(M_PI * V * t);
}

// ============================= Kernels ======================================
// Shared-memory first step (2nd-order Taylor): u1 = u0 + 0.5*r^2 * (u0[j+1] - 2u0[j] + u0[j-1])
// Computes interior indices i in [1, N-2].
__global__ void first_step_kernel_shmem(double* __restrict__ u1, const double* __restrict__ u0,int N, double r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // interior start at 1
    if (i >= N - 1) return;

    extern __shared__ double s[]; // size = blockDim.x + 2
    int t = threadIdx.x;

    // center
    s[t + 1] = u0[i];
    // halos
    if (t == 0)                         s[0]          = u0[i - 1];
    if (t == blockDim.x - 1 || i == N-2) s[t + 2]      = u0[i + 1];

    __syncthreads();

    double lap = s[t + 2] - 2.0 * s[t + 1] + s[t];
    u1[i] = u0[i] + 0.5 * r2 * lap;
}

// Shared-memory leapfrog: u_new = 2 u_now - u_old + r^2 * Δxx u_now, interior only
__global__ void leapfrog_kernel_shmem(double* __restrict__ u_new,
    const double* __restrict__ u_now,
    const double* __restrict__ u_old,
    int N, double r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // interior
    if (i >= N - 1) return;

    extern __shared__ double s[]; // size = blockDim.x + 2
    int t = threadIdx.x;

    s[t + 1] = u_now[i];
    if (t == 0)                         s[0]          = u_now[i - 1];
    if (t == blockDim.x - 1 || i == N-2) s[t + 2]      = u_now[i + 1];

    __syncthreads();

    double lap = s[t + 2] - 2.0 * s[t + 1] + s[t];
    u_new[i] = 2.0 * u_now[i] - u_old[i] + r2 * lap;
}

// Optional: per-block partial reductions for L1, L2, and ||u_exact||^2 (to form relL2)
// No atomics; each block writes one triple to partial arrays.
#if USE_GPU_ERROR
__global__ void error_partials_kernel(
    const double* __restrict__ u,
    int N, double dx, double t,
    double* __restrict__ partL1,
    double* __restrict__ partL2,
    double* __restrict__ partUE2)
{
    int tid    = threadIdx.x;
    int gid    = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sL1 = 0.0, sL2 = 0.0, sUE2 = 0.0;
    for (int j = gid; j < N; j += stride) {
        double x   = j * dx;
        double ue  = exact(x, t);
        double diff = u[j] - ue;
        sL1 += fabs(diff) * dx;
        sL2 += diff * diff * dx;
        sUE2 += ue * ue * dx;
    }

    extern __shared__ double smem[];
    double* bufL1 = smem;                      // blockDim.x
    double* bufL2 = smem + blockDim.x;         // blockDim.x
    double* bufUE = smem + 2 * blockDim.x;     // blockDim.x

    bufL1[tid] = sL1;
    bufL2[tid] = sL2;
    bufUE[tid] = sUE2;
    __syncthreads();

    // parallel reduction in shared memory
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            bufL1[tid] += bufL1[tid + offset];
            bufL2[tid] += bufL2[tid + offset];
            bufUE[tid] += bufUE[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partL1[blockIdx.x] = bufL1[0];
        partL2[blockIdx.x] = bufL2[0];
        partUE2[blockIdx.x] = bufUE[0];
    }
}
#endif // USE_GPU_ERROR

// ============================= Host helpers =================================
static void initialize_host(std::vector<double>& u0, int N, double dx) {
    for (int j = 0; j < N; ++j) {
        double x = j * dx;
        u0[j] = sin(M_PI * x); // initial displacement
    }
    u0[0] = 0.0; u0[N - 1] = 0.0; // Dirichlet boundaries
}

static void compute_errors_cpu(const std::vector<double>& u, int N, double dx, double t,
                               double& L1, double& L2, double& relL2)
{
    double e1 = 0.0, e2 = 0.0, n2 = 0.0;
    for (int j = 0; j < N; ++j) {
        double x = j * dx;
        double ue = exact(x, t);
        double diff = u[j] - ue;
        e1 += std::abs(diff) * dx;
        e2 += diff * diff * dx;
        n2 += ue * ue * dx;
    }
    L1 = e1;
    L2 = std::sqrt(e2);
    relL2 = (n2 > 0) ? std::sqrt(e2 / n2) : 0.0;
}

// Ensure boundaries are zero on device arrays (tiny kernel-free approach via Memset for u1/u2)
static inline void zero_device(double* d_ptr, size_t bytes, cudaStream_t s) {
    CUDA_CHECK(cudaMemsetAsync(d_ptr, 0, bytes, s));
}

// ================================ main ======================================
int main() {
    // Output files
    FILE* ferr = std::fopen("errors.csv",  "w");
    FILE* ftim = std::fopen("timings.csv", "w");
    if (!ferr || !ftim) { std::fprintf(stderr, "open output failed\n"); return 1; }
    std::fprintf(ferr, "N,dx,dt,l1_error,l2_error,relative_l2\n");
    std::fprintf(ftim, "N,h2d_ms,kernel_ms,d2h_ms,total_ms\n");

    //the best perf on the GPU.
    const int blockSize = 256;

    // Precompute maximum N and corresponding maximum grid size used by interior-point kernels
    int maxN = 0; for (int i = 0; i < nCases; ++i) maxN = std::max(maxN, Ns[i]);
    int maxInterior = std::max(0, maxN - 2);
    int gridMax = (maxInterior + blockSize - 1) / blockSize;
    if (gridMax < 1) gridMax = 1; // safety

    // Device buffers (reused across cases)
    double *d_u0 = nullptr, *d_u1 = nullptr, *d_u2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u0, maxN * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u1, maxN * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u2, maxN * sizeof(double)));

#if USE_GPU_ERROR
    // Partials for error reductions (capacity = gridMax)
    double *d_partL1 = nullptr, *d_partL2 = nullptr, *d_partUE2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partL1,  gridMax * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partL2,  gridMax * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partUE2, gridMax * sizeof(double)));
    std::vector<double> h_partL1(gridMax), h_partL2(gridMax), h_partUE2(gridMax);
#endif

    // Host buffers
#if USE_PINNED_HOST
    double *h_u0 = nullptr, *h_u1 = nullptr; // we only need two host buffers here
    CUDA_CHECK(cudaHostAlloc(&h_u0, maxN * sizeof(double), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_u1, maxN * sizeof(double), cudaHostAllocPortable));
    std::vector<double> h_u1_vec; // only used if we want CPU error without pinned copy
#else
    std::vector<double> h_u0(maxN), h_u1(maxN);
#endif

    // Stream & events (reused across runs)
    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
    cudaEvent_t ev_h2d_start, ev_h2d_stop;
    cudaEvent_t ev_k_start,  ev_k_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&ev_k_start));
    CUDA_CHECK(cudaEventCreate(&ev_k_stop));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

    // (Optional) Favor shared memory for our stencil kernels
    CUDA_CHECK(cudaFuncSetCacheConfig(first_step_kernel_shmem, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetCacheConfig(leapfrog_kernel_shmem,   cudaFuncCachePreferShared));

    for (int ci = 0; ci < nCases; ++ci) {
        const int N   = Ns[ci];
        const double dx = 1.0 / (N - 1);
        double dt = CFL * dx / V;

        int steps = (int)floor(Tfinal / dt + 1e-12);
        if (steps < 1) steps = 1;
        dt = Tfinal / steps; // adjust dt to hit Tfinal exactly
        const double t_final = steps * dt;

        const double r  = V * dt / dx;
        const double r2 = r * r;

        // Prepare host u0 and copy to device
#if USE_PINNED_HOST
        std::vector<double> htmp(N);
        initialize_host(htmp, N, dx);
        // copy into pinned buffer (htmp only helps use vector math if you prefer)
        std::memcpy(h_u0, htmp.data(), N * sizeof(double));
#else
        initialize_host(h_u0, N, dx);
#endif

        // Compute grid for interior points
        int interior = std::max(0, N - 2);
        int gridSize = (interior + blockSize - 1) / blockSize;
        if (gridSize < 1) gridSize = 1;
        size_t shmem_bytes = (size_t)(blockSize + 2) * sizeof(double);

        // --- H2D timing ---
        CUDA_CHECK(cudaEventRecord(ev_h2d_start, s));
        CUDA_CHECK(cudaMemcpyAsync(d_u0, h_u0, N * sizeof(double), cudaMemcpyHostToDevice, s));
        // Ensure u1/u2 are zero for boundaries (and interior before first write)
        zero_device(d_u1, N * sizeof(double), s);
        zero_device(d_u2, N * sizeof(double), s);
        CUDA_CHECK(cudaEventRecord(ev_h2d_stop, s));
        CUDA_CHECK(cudaEventSynchronize(ev_h2d_stop));
        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_stop));

        // --- Kernel timing: solver only (first step + leapfrog steps) ---
        CUDA_CHECK(cudaEventRecord(ev_k_start, s));

        first_step_kernel_shmem<<<gridSize, blockSize, shmem_bytes, s>>>(d_u1, d_u0, N, r2);
#if ENABLE_KERNEL_CHECKS
        cudaPeekAtLastError();
#endif
        for (int step = 1; step < steps; ++step) {
            leapfrog_kernel_shmem<<<gridSize, blockSize, shmem_bytes, s>>>(d_u2, d_u1, d_u0, N, r2);
#if ENABLE_KERNEL_CHECKS
            cudaPeekAtLastError();
#endif
            // rotate pointers
            double* tmp = d_u0; d_u0 = d_u1; d_u1 = d_u2; d_u2 = tmp;
        }

        CUDA_CHECK(cudaEventRecord(ev_k_stop, s));
        CUDA_CHECK(cudaEventSynchronize(ev_k_stop));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_k_start, ev_k_stop));

        // --- GPU error reduction (optional) ---
        double L1 = 0.0, L2 = 0.0, relL2 = 0.0;
#if USE_GPU_ERROR
        // Launch with same grid/block; shared memory = 3 * blockSize * sizeof(double)
        size_t shmem_err = (size_t)3 * blockSize * sizeof(double);
        error_partials_kernel<<<gridSize, blockSize, shmem_err, s>>>(d_u1, N, dx, t_final,
                                                                     d_partL1, d_partL2, d_partUE2);
#if ENABLE_KERNEL_CHECKS
        cudaPeekAtLastError();
#endif
        CUDA_CHECK(cudaMemcpyAsync(h_partL1.data(), d_partL1, gridSize * sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h_partL2.data(), d_partL2, gridSize * sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h_partUE2.data(), d_partUE2, gridSize * sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        double sumL1 = 0.0, sumL2 = 0.0, sumUE2 = 0.0;
        for (int k = 0; k < gridSize; ++k) { sumL1 += h_partL1[k]; sumL2 += h_partL2[k]; sumUE2 += h_partUE2[k]; }
        L1 = sumL1; L2 = std::sqrt(sumL2); relL2 = (sumUE2 > 0) ? std::sqrt(sumL2 / sumUE2) : 0.0;
#else
        // If USE_GPU_ERROR=0:
        // compute_errors_cpu(h_u1_vec, N, dx, t_final, L1, L2, relL2);
#endif

        // --- D2H timing (fetch full profile for CSV) ---
        // Note: If you don't need profile_Nxxx.csv, you can skip this full copy and only copy small summaries.
        CUDA_CHECK(cudaEventRecord(ev_d2h_start, s));
#if USE_PINNED_HOST
        CUDA_CHECK(cudaMemcpyAsync(h_u1, d_u1, N * sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaEventRecord(ev_d2h_stop, s));
        CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));
        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));
#else
        CUDA_CHECK(cudaMemcpyAsync(h_u1.data(), d_u1, N * sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaEventRecord(ev_d2h_stop, s));
        CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));
        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));
#endif

        float total_ms = h2d_ms + kernel_ms + d2h_ms;

        // --- Write CSV summaries ---
        std::fprintf(ferr, "%d,%.8f,%.8f,%.10e,%.10e,%.10e\n", N, dx, dt, L1, L2, relL2);
        std::fprintf(ftim, "%d,%.3f,%.3f,%.3f,%.3f\n", N, h2d_ms, kernel_ms, d2h_ms, total_ms);

        // --- Write profile for visual check (x, u_ex, u_num) ---
        char pname[64];
        std::snprintf(pname, sizeof(pname), "profile_N%03d.csv", N);
        if (FILE* pf = std::fopen(pname, "w")) {
            std::fprintf(pf, "x,u_ex,u_num\n");
#if USE_PINNED_HOST
            for (int j = 0; j < N; ++j) {
                double x  = j * dx;
                double ue = exact(x, t_final);
                std::fprintf(pf, "%.8f,%.8f,%.8f\n", x, ue, h_u1[j]);
            }
#else
            for (int j = 0; j < N; ++j) {
                double x  = j * dx;
                double ue = exact(x, t_final);
                std::fprintf(pf, "%.8f,%.8f,%.8f\n", x, ue, h_u1[j]);
            }
#endif
            std::fclose(pf);
        }

        std::printf("[N=%5d] steps=%d, dx=%.6g, dt=%.6g | L2=%.3e, relL2=%.3e | H2D=%.2f ms, K=%.2f ms, D2H=%.2f ms, TOT=%.2f ms\n",
                    N, steps, dx, dt, L2, relL2, h2d_ms, kernel_ms, d2h_ms, total_ms);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_h2d_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_stop));
    CUDA_CHECK(cudaEventDestroy(ev_k_start));
    CUDA_CHECK(cudaEventDestroy(ev_k_stop));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_start));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_stop));
    CUDA_CHECK(cudaStreamDestroy(s));

#if USE_GPU_ERROR
    CUDA_CHECK(cudaFree(d_partL1));
    CUDA_CHECK(cudaFree(d_partL2));
    CUDA_CHECK(cudaFree(d_partUE2));
#endif

    CUDA_CHECK(cudaFree(d_u0));
    CUDA_CHECK(cudaFree(d_u1));
    CUDA_CHECK(cudaFree(d_u2));

#if USE_PINNED_HOST
    CUDA_CHECK(cudaFreeHost(h_u0));
    CUDA_CHECK(cudaFreeHost(h_u1));
#endif

    std::fclose(ferr);
    std::fclose(ftim);
    std::puts("Done. Wrote errors.csv, timings.csv and profile_Nxxx.csv");
    return 0;
}
