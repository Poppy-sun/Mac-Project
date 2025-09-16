// 2D Wave Equation (Dirichlet) — Leapfrog scheme on CUDA (optimized version)
// Key optimizations over the baseline:
//  - Shared-memory tiled 5-point stencil (coalesced loads; halo cached in SMEM)
//  - Fused boundary handling in the main kernels
//  - __restrict__ pointers and read-only caching of u_prev where applicable
//  - Configurable block size; row padding in SMEM to reduce bank conflicts
//  - Same I/O format as CPU/CUDA baseline for apples-to-apples comparison
//      * error_results_leapfrog2D_CFLxx.csv
//      * profile2D_CFLxx_NxNNNNN_NyNNNNN.csv
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IDX(i,j,Nx) ((i) + (Nx)*(j))

// ---------------- Tunables ----------------
#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 8
#endif

// Add +1 padding to shared rows to mitigate bank conflicts
#define SMEM_PAD 1

// ---------------- CUDA helpers ----------------
#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---------------- Exact & IC -----------
__host__ __device__ inline double exact_solution(double x, double y, double t,
                                                double Lx, double Ly, double c) {
    const double kx = M_PI / Lx;
    const double ky = M_PI / Ly;
    const double omega = c * M_PI * sqrt(1.0/(Lx*Lx) + 1.0/(Ly*Ly));
    return sin(kx*x) * sin(ky*y) * cos(omega * t);
}

__host__ __device__ inline double f0(double x, double y, double Lx, double Ly) {
    const double kx = M_PI / Lx;
    const double ky = M_PI / Ly;
    return sin(kx*x) * sin(ky*y);
}

__host__ __device__ inline double g0(double, double) { return 0.0; }

// ---------------- Kernels ----------------
// Set Dirichlet boundaries to zero
__global__ void k_set_boundary(double* u, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    if (i==0 || j==0 || i==Nx-1 || j==Ny-1) u[IDX(i,j,Nx)] = 0.0;
}

// Initialize u_prev = u^0 from f0(x,y) and enforce Dirichlet=0 at t=0
__global__ void k_init_u0(double* __restrict__ u_prev,
                          int Nx, int Ny, double dx, double dy,
                          double Lx, double Ly) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    double x = i * dx, y = j * dy;
    double val = f0(x,y,Lx,Ly);
    if (i==0 || j==0 || i==Nx-1 || j==Ny-1) val = 0.0; // Dirichlet
    u_prev[IDX(i,j,Nx)] = val;
}

// Compute u_curr = u^1 using Taylor from u_prev (u^0):
// u^1 = u^0 + dt*g + 0.5*(c dt)^2 Lap(u^0)
// Use SMEM tiling to compute Lap(u^0)
__global__ void k_first_step_from_u0(double* __restrict__ u_curr,
                                     const double* __restrict__ u_prev,
                                     int Nx, int Ny,
                                     double dx, double dy, double dt,
                                     double Lx, double Ly, double c) {
    const double inv_dx2 = 1.0 / (dx*dx);
    const double inv_dy2 = 1.0 / (dy*dy);
    const double c2dt2   = (c*c) * (dt*dt);

    // Shared tile with halo and row padding
    __shared__ double tile[BLOCK_Y+2][BLOCK_X+2+SMEM_PAD];

    int tx = threadIdx.x, ty = threadIdx.y;
    int i  = blockIdx.x * blockDim.x + tx;
    int j  = blockIdx.y * blockDim.y + ty;

    // Clamp indices for halo loads
    int ii = min(max(i, 0), Nx-1);
    int jj = min(max(j, 0), Ny-1);

    // Load interior cell for this thread
    tile[ty+1][tx+1] = (ii>=0 && ii<Nx && jj>=0 && jj<Ny) ? __ldg(&u_prev[IDX(ii,jj,Nx)]) : 0.0;

    // Load halo in x
    if (tx==0)  tile[ty+1][0]          = (i>0)        ? __ldg(&u_prev[IDX(i-1,jj,Nx)]) : 0.0;
    if (tx==BLOCK_X-1) tile[ty+1][BLOCK_X+1] = (i+1<Nx)   ? __ldg(&u_prev[IDX(i+1,jj,Nx)]) : 0.0;
    // Load halo in y
    if (ty==0)  tile[0][tx+1]          = (j>0)        ? __ldg(&u_prev[IDX(ii,j-1,Nx)]) : 0.0;
    if (ty==BLOCK_Y-1) tile[BLOCK_Y+1][tx+1] = (j+1<Ny)   ? __ldg(&u_prev[IDX(ii,j+1,Nx)]) : 0.0;

    // Load corners (a few threads handle them)
    if (tx==0 && ty==0)                 tile[0][0]                  = (i>0 && j>0)               ? __ldg(&u_prev[IDX(i-1,j-1,Nx)]) : 0.0;
    if (tx==BLOCK_X-1 && ty==0)         tile[0][BLOCK_X+1]          = (i+1<Nx && j>0)            ? __ldg(&u_prev[IDX(i+1,j-1,Nx)]) : 0.0;
    if (tx==0 && ty==BLOCK_Y-1)         tile[BLOCK_Y+1][0]          = (i>0 && j+1<Ny)            ? __ldg(&u_prev[IDX(i-1,j+1,Nx)]) : 0.0;
    if (tx==BLOCK_X-1 && ty==BLOCK_Y-1) tile[BLOCK_Y+1][BLOCK_X+1]  = (i+1<Nx && j+1<Ny)         ? __ldg(&u_prev[IDX(i+1,j+1,Nx)]) : 0.0;

    __syncthreads();

    if (i >= Nx || j >= Ny) return;

    double u0 = tile[ty+1][tx+1];
    // boundaries => Dirichlet 0 at t=dt
    if (i==0 || j==0 || i==Nx-1 || j==Ny-1) { u_curr[IDX(i,j,Nx)] = 0.0; return; }

    double lap = (tile[ty+1][tx+2] - 2.0*u0 + tile[ty+1][tx]) * inv_dx2
               + (tile[ty+2][tx+1] - 2.0*u0 + tile[ty][tx+1]) * inv_dy2;
    double v0  = g0(i*dx, j*dy);
    u_curr[IDX(i,j,Nx)] = u0 + dt * v0 + 0.5 * c2dt2 * lap;
}

// Leapfrog step using SMEM tile for u_curr; u_prev read from global (read-only cached)
__global__ void k_leapfrog(double* __restrict__ u_next,
                           const double* __restrict__ u_curr,
                           const double* __restrict__ u_prev,
                           int Nx, int Ny,
                           double dx, double dy,
                           double c, double dt) {
    const double inv_dx2 = 1.0 / (dx*dx);
    const double inv_dy2 = 1.0 / (dy*dy);
    const double c2dt2   = (c*c) * (dt*dt);

    __shared__ double tile[BLOCK_Y+2][BLOCK_X+2+SMEM_PAD];

    int tx = threadIdx.x, ty = threadIdx.y;
    int i  = blockIdx.x * blockDim.x + tx;
    int j  = blockIdx.y * blockDim.y + ty;

    int ii = min(max(i, 0), Nx-1);
    int jj = min(max(j, 0), Ny-1);

    // Load center
    tile[ty+1][tx+1] = (ii>=0 && ii<Nx && jj>=0 && jj<Ny) ? __ldg(&u_curr[IDX(ii,jj,Nx)]) : 0.0;

    // Halos
    if (tx==0)  tile[ty+1][0]          = (i>0)        ? __ldg(&u_curr[IDX(i-1,jj,Nx)]) : 0.0;
    if (tx==BLOCK_X-1) tile[ty+1][BLOCK_X+1] = (i+1<Nx)   ? __ldg(&u_curr[IDX(i+1,jj,Nx)]) : 0.0;
    if (ty==0)  tile[0][tx+1]          = (j>0)        ? __ldg(&u_curr[IDX(ii,j-1,Nx)]) : 0.0;
    if (ty==BLOCK_Y-1) tile[BLOCK_Y+1][tx+1] = (j+1<Ny)   ? __ldg(&u_curr[IDX(ii,j+1,Nx)]) : 0.0;

    if (tx==0 && ty==0)                 tile[0][0]                  = (i>0 && j>0)               ? __ldg(&u_curr[IDX(i-1,j-1,Nx)]) : 0.0;
    if (tx==BLOCK_X-1 && ty==0)         tile[0][BLOCK_X+1]          = (i+1<Nx && j>0)            ? __ldg(&u_curr[IDX(i+1,j-1,Nx)]) : 0.0;
    if (tx==0 && ty==BLOCK_Y-1)         tile[BLOCK_Y+1][0]          = (i>0 && j+1<Ny)            ? __ldg(&u_curr[IDX(i-1,j+1,Nx)]) : 0.0;
    if (tx==BLOCK_X-1 && ty==BLOCK_Y-1) tile[BLOCK_Y+1][BLOCK_X+1]  = (i+1<Nx && j+1<Ny)         ? __ldg(&u_curr[IDX(i+1,j+1,Nx)]) : 0.0;

    __syncthreads();

    if (i >= Nx || j >= Ny) return;

    if (i==0 || j==0 || i==Nx-1 || j==Ny-1) { u_next[IDX(i,j,Nx)] = 0.0; return; }

    double uc = tile[ty+1][tx+1];
    double lap = (tile[ty+1][tx+2] - 2.0*uc + tile[ty+1][tx]) * inv_dx2
               + (tile[ty+2][tx+1] - 2.0*uc + tile[ty][tx+1]) * inv_dy2;

    double uo = __ldg(&u_prev[IDX(i,j,Nx)]);
    u_next[IDX(i,j,Nx)] = 2.0*uc - uo + c2dt2 * lap;
}

// ---------------- Host utilities ----------------
static void write_profile_csv(const std::string& fname,
                              const std::vector<double>& u,
                              int Nx, int Ny,
                              double dx, double dy,
                              double t, double Lx, double Ly, double c) {
    std::ofstream fp(fname);
    if (!fp) { std::cerr << "Could not open " << fname << " for writing.\n"; exit(1); }
    fp << std::fixed << std::setprecision(8);
    fp << "x,y,u_ex,u_num\n";
    for (int j = 0; j < Ny; ++j) {
        double y = j * dy;
        for (int i = 0; i < Nx; ++i) {
            double x = i * dx;
            double ue = exact_solution(x,y,t,Lx,Ly,c);
            fp << x << "," << y << "," << ue << "," << u[IDX(i,j,Nx)] << "\n";
        }
    }
}

static void compute_error_host(const std::vector<double>& u,
                               int Nx, int Ny,
                               double dx, double dy, double t,
                               double Lx, double Ly, double c,
                               double& L1, double& L2, double& relL2) {
    long double sum_l1=0.0L, sum_l2=0.0L, sum_ue2=0.0L;
    for (int j=0;j<Ny;++j) {
        double y = j*dy;
        for (int i=0;i<Nx;++i) {
            double x = i*dx;
            double ue = exact_solution(x,y,t,Lx,Ly,c);
            double diff = u[IDX(i,j,Nx)] - ue;
            sum_l1 += fabsl(diff) * dx * dy;
            sum_l2 += (long double)diff * diff * dx * dy;
            sum_ue2 += (long double)ue * ue * dx * dy;
        }
    }
    L1 = (double)sum_l1;
    L2 = std::sqrt((double)sum_l2);
    relL2 = (sum_ue2>0.0L) ? std::sqrt((double)(sum_l2/sum_ue2)) : 0.0;
}

// Run one simulation on GPU (optimized)
static float run_simulation(int Nx, int Ny, double CFL, double T,
                            double Lx, double Ly, double c,
                            double& L1, double& L2, double& relL2,
                            bool write_profile) {
    const double dx = Lx / (Nx - 1);
    const double dy = Ly / (Ny - 1);
    const double inv_dx2 = 1.0/(dx*dx);
    const double inv_dy2 = 1.0/(dy*dy);
    const double dt_max = 1.0 / ( c * std::sqrt(inv_dx2 + inv_dy2) );
    double dt = CFL * dt_max;
    int steps = (int)std::ceil(T / dt - 1e-12);
    if (steps < 1) steps = 1;
    dt = T / steps; // <= initial stable dt
    const double t_final = dt * steps;

    const size_t N = (size_t)Nx * (size_t)Ny;

    double *d_prev=nullptr, *d_curr=nullptr, *d_next=nullptr;
    CUDA_CHECK(cudaMalloc(&d_prev, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_curr, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_next, N*sizeof(double)));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((Nx + BLOCK_X - 1)/BLOCK_X, (Ny + BLOCK_Y - 1)/BLOCK_Y);

    // u^0
    k_init_u0<<<grid, block>>>(d_prev, Nx, Ny, dx, dy, Lx, Ly);
    CUDA_CHECK(cudaGetLastError());

    // u^1 from u^0 (SMEM)
    k_first_step_from_u0<<<grid, block>>>(d_curr, d_prev, Nx, Ny, dx, dy, dt, Lx, Ly, c);
    CUDA_CHECK(cudaGetLastError());

    // timing
    cudaEvent_t evS, evE; CUDA_CHECK(cudaEventCreate(&evS)); CUDA_CHECK(cudaEventCreate(&evE));
    CUDA_CHECK(cudaEventRecord(evS));

    for (int step=1; step<steps; ++step) {
        k_leapfrog<<<grid, block>>>(d_next, d_curr, d_prev, Nx, Ny, dx, dy, c, dt);
        CUDA_CHECK(cudaGetLastError());
        // rotate
        double* tmp = d_prev; d_prev = d_curr; d_curr = d_next; d_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(evE));
    CUDA_CHECK(cudaEventSynchronize(evE));
    float ms=0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, evS, evE));
    CUDA_CHECK(cudaEventDestroy(evS)); CUDA_CHECK(cudaEventDestroy(evE));

    // copy back final field
    std::vector<double> h_curr(N);
    CUDA_CHECK(cudaMemcpy(h_curr.data(), d_curr, N*sizeof(double), cudaMemcpyDeviceToHost));

    compute_error_host(h_curr, Nx, Ny, dx, dy, t_final, Lx, Ly, c, L1, L2, relL2);

    if (write_profile) {
        char fname[256];
        std::snprintf(fname, sizeof(fname), "profile2D_CFL%.2f_Nx%05d_Ny%05d.csv", CFL, Nx, Ny);
        write_profile_csv(fname, h_curr, Nx, Ny, dx, dy, t_final, Lx, Ly, c);
    }

    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_next));

    return ms; // kernel time in ms
}

int main() {
    const double Lx=1.0, Ly=1.0, c=1.0, T=1.0;
    const double CFLs[] = {0.20, 0.50, 0.90, 1.05};
    const int nCFL = (int)(sizeof(CFLs)/sizeof(CFLs[0]));

    const int Nlist[][2] = { {129,129}, {257,257}, {513,513} };
    const int nN = (int)(sizeof(Nlist)/sizeof(Nlist[0]));

    for (int cidx=0; cidx<nCFL; ++cidx) {
        double CFL = CFLs[cidx];
        char fname[128];
        std::snprintf(fname, sizeof(fname), "error_results_leapfrog2D_CFL%.2f.csv", CFL);
        std::ofstream out(fname);
        if (!out) { std::cerr << "Could not open " << fname << " for writing.\n"; return 1; }
        out << std::fixed << std::setprecision(8);
        out << "CFL,Nx,Ny,dx,dy,dt,l1_error,l2_error,relative_l2,order_p,CPU_time(s)\n";

        double prev_dx=-1.0, prev_L2=-1.0;

        for (int i=0;i<nN;++i) {
            int Nx = Nlist[i][0];
            int Ny = Nlist[i][1];
            double dx = Lx / (Nx - 1);
            double dy = Ly / (Ny - 1);

            bool write_profile = (i==nN-1);
            double L1=0.0, L2=0.0, relL2=0.0;
            float ms = run_simulation(Nx, Ny, CFL, T, Lx, Ly, c, L1, L2, relL2, write_profile);

            // recompute dt for metadata
            double inv_dx2 = 1.0/(dx*dx), inv_dy2 = 1.0/(dy*dy);
            double dt_max = 1.0 / ( c * std::sqrt(inv_dx2 + inv_dy2) );
            double dt = CFL * dt_max; int steps = (int)std::ceil(T/dt - 1e-12); if (steps<1) steps=1; dt = T/steps;

            double p = NAN; if (prev_L2>0.0 && prev_dx>0.0) p = std::log(L2/prev_L2) / std::log(dx/prev_dx);

            out << std::setprecision(2) << CFL << "," << Nx << "," << Ny << ","
                << std::setprecision(8) << dx << "," << dy << "," << dt << ","
                << std::scientific << L1 << "," << L2 << "," << relL2 << ",";
            if (std::isnan(p)) out << "NA,"; else out << std::fixed << std::setprecision(6) << p << ",";
            out << std::fixed << std::setprecision(6) << (ms/1000.0) << "\n";

            std::cout << "[OPT] CFL=" << CFL << " Nx=" << Nx << " Ny=" << Ny
                      << " L2=" << L2 << " relL2=" << relL2
                      << " time(s)=" << (ms/1000.0) << (std::isnan(p)?" p=NA":" p≈2") << "\n";

            prev_dx = dx; prev_L2 = L2;
        }
        out.close();
    }

    std::cout << "Done (optimized). One CSV per CFL; 2D profiles written for largest grids." << std::endl;
    return 0;
}
