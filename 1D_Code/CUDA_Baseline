// wave_gpu.cu  — Leapfrog (1D wave) CUDA Baseline implementation + split time
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define ENABLE_KERNEL_CHECKS 1

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// parameters
static constexpr double V       = 1.0;        // wave speed
static constexpr double CFL     = 0.50;       // CFL number
static constexpr double Tfinal  = 10.23;       //superconvergence at T=1,T=10

//Ns
static const int Ns[] = {4096,8192,16384,32768};
static const int nCases = int(sizeof(Ns)/sizeof(Ns[0]));

#define CUDA_CHECK(call) do {                                        
    cudaError_t _e = (call);                                         
    if (_e != cudaSuccess) {                                         
        fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",             
                cudaGetErrorString(_e), _e, __FILE__, __LINE__);     
        exit(1);                                                     
    }                                                                
} while(0)

// ===================exact solution (host + device)=============
__host__ __device__ inline double exact(double x, double t){
    return sin(M_PI * x) * cos(M_PI * V * t);
}

// ========================Kernel function:=====================
// First step using second order Tylor expansion
__global__ void first_step_kernel(double* u1, const double* u0, int N, double r2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    if (j == 0 || j == N-1) { u1[j] = 0.0; return; }
    u1[j] = u0[j] + 0.5 * r2 * (u0[j+1] - 2.0*u0[j] + u0[j-1]);
}

// Leapfrog：u^{n+1} = 2u^n - u^{n-1} + r^2 Δxx u^n，Dirichlet boundary 0
__global__ void leapfrog_kernel(double* u_new, const double* u_now, const double* u_old, int N, double r2){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    if (j == 0 || j == N-1) { u_new[j] = 0.0; return; }
    u_new[j] = 2.0*u_now[j] - u_old[j] + r2*(u_now[j+1] - 2.0*u_now[j] + u_now[j-1]);
}

// Host dunction:
static void initialize(std::vector<double>& u0, int N, double dx){
    for (int j=0; j<N; ++j){
        double x = j*dx;
        u0[j] = sin(M_PI * x);   // initial position
    }
    u0[0] = 0.0; u0[N-1] = 0.0;  // Dirichlet
}

static void compute_errors(const std::vector<double>& u, int N, double dx, double t,
                           double& L1, double& L2, double& relL2){
    double e1=0.0, e2=0.0, n2=0.0;
    for (int j=0; j<N; ++j){
        double x = j*dx;
        double ue = exact(x, t);
        double diff = u[j] - ue;
        e1 += std::abs(diff) * dx;  
        e2 += diff*diff * dx;       
        n2 += ue*ue * dx;
    }
    L1 = e1;
    L2 = std::sqrt(e2);
    relL2 = (n2 > 0) ? std::sqrt(e2/n2) : 0.0;
}

int main(){
    // Output files:
    FILE* ferr = std::fopen("errors.csv",  "w");
    FILE* ftim = std::fopen("timings.csv", "w");
    if (!ferr || !ftim) { std::fprintf(stderr, "open output failed\n"); return 1; }
    std::fprintf(ferr, "N,dx,dt,l1_error,l2_error,relative_l2\n");
    std::fprintf(ftim, "N,h2d_ms,kernel_ms,d2h_ms,total_ms\n");

    for (int ci=0; ci<nCases; ++ci){
        int N   = Ns[ci];
        double dx = 1.0 / (N - 1);
        double dt = CFL * dx / V;

        int steps = (int)floor(Tfinal / dt + 1e-12);
        if (steps < 1) steps = 1;
        dt = Tfinal / steps;                
        double t_final = steps * dt;

        double r  = V * dt / dx;
        double r2 = r * r;

        // Host :
        std::vector<double> h_u0(N), h_u1(N), h_u2(N);
        initialize(h_u0, N, dx);

        // Device :
        double *d_u0=nullptr, *d_u1=nullptr, *d_u2=nullptr;
        CUDA_CHECK(cudaMalloc(&d_u0, N*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_u1, N*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_u2, N*sizeof(double)));

        //Timing
        cudaEvent_t ev_h2d_start, ev_h2d_stop;
        cudaEvent_t ev_k_start,  ev_k_stop;
        cudaEvent_t ev_d2h_start, ev_d2h_stop;
        CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
        CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
        CUDA_CHECK(cudaEventCreate(&ev_k_start));
        CUDA_CHECK(cudaEventCreate(&ev_k_stop));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

        // ---- H2D 计时 ----
        CUDA_CHECK(cudaEventRecord(ev_h2d_start, 0));
        CUDA_CHECK(cudaMemcpy(d_u0, h_u0.data(), N*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_h2d_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_h2d_stop));
        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_stop));

        // kernel :
        int blockSize = 256;
        int gridSize  = (N + blockSize - 1) / blockSize;

        // Kernel timing:
        CUDA_CHECK(cudaEventRecord(ev_k_start, 0));

        first_step_kernel<<<gridSize, blockSize>>>(d_u1, d_u0, N, r2);
        #if defined(ENABLE_KERNEL_CHECKS)
        cudaPeekAtLastError();
        #endif

        for (int s=1; s<steps; ++s){
            leapfrog_kernel<<<gridSize, blockSize>>>(d_u2, d_u1, d_u0, N, r2);
        # if defined(ENABLE_KERNEL_CHECKS)
        cudaPeekAtLastError();
        #endif
    
            double* tmp = d_u0; d_u0 = d_u1; d_u1 = d_u2; d_u2 = tmp;
        }

        CUDA_CHECK(cudaEventRecord(ev_k_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_k_stop));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_k_start, ev_k_stop));

        // D2H timing
        CUDA_CHECK(cudaEventRecord(ev_d2h_start, 0));
        CUDA_CHECK(cudaMemcpy(h_u1.data(), d_u1, N*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev_d2h_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));
        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));

        float total_ms = h2d_ms + kernel_ms + d2h_ms;

        // Errors:
        double L1=0, L2=0, rL2=0;
        compute_errors(h_u1, N, dx, t_final, L1, L2, rL2);

        //Output files
        std::fprintf(ferr, "%d,%.8f,%.8f,%.10e,%.10e,%.10e\n",
                     N, dx, dt, L1, L2, rL2);
        std::fprintf(ftim, "%d,%.3f,%.3f,%.3f,%.3f\n",
                     N, h2d_ms, kernel_ms, d2h_ms, total_ms);

        //Output profile (easy "numerical vs analytical" comparison)
        char pname[64];
        std::snprintf(pname, sizeof(pname), "profile_N%03d.csv", N);
        if (FILE* pf = std::fopen(pname, "w")){
            std::fprintf(pf, "x,u_ex,u_num\n");
            for (int j=0; j<N; ++j){
                double x  = j*dx;
                double ue = exact(x, t_final);
                std::fprintf(pf, "%.8f,%.8f,%.8f\n", x, ue, h_u1[j]);
            }
            std::fclose(pf);
        }

        
        std::printf("[N=%4d] steps=%d, dx=%.6g, dt=%.6g | L2=%.3e, relL2=%.3e | H2D=%.2f ms, K=%.2f ms, D2H=%.2f ms, TOT=%.2f ms\n",
                    N, steps, dx, dt, L2, rL2, h2d_ms, kernel_ms, d2h_ms, total_ms);

        // release
        CUDA_CHECK(cudaEventDestroy(ev_h2d_start));
        CUDA_CHECK(cudaEventDestroy(ev_h2d_stop));
        CUDA_CHECK(cudaEventDestroy(ev_k_start));
        CUDA_CHECK(cudaEventDestroy(ev_k_stop));
        CUDA_CHECK(cudaEventDestroy(ev_d2h_start));
        CUDA_CHECK(cudaEventDestroy(ev_d2h_stop));
        CUDA_CHECK(cudaFree(d_u0));
        CUDA_CHECK(cudaFree(d_u1));
        CUDA_CHECK(cudaFree(d_u2));
    }

    std::fclose(ferr);
    std::fclose(ftim);
    std::puts("Done. Wrote errors.csv, timings.csv and profile_Nxxx.csv");
    return 0;
}
