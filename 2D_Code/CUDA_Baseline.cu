// 2D Wave Equation (Dirichlet) — Leapfrog scheme on CUDA (baseline)
// - Standing-wave IC with analytical solution for verification
// - Double precision
// - Simple single-GPU baseline 


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IDX(i,j,Nx) ((i) + (Nx)*(j))

// ---------------- CUDA error helper ----------------
#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---------------- exact / IC (device + host) ----------------
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

// ---------------- kernels ----------------
__global__ void kernel_set_zero_boundary(double* u, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    if (i==0 || i==Nx-1 || j==0 || j==Ny-1) {
        u[IDX(i,j,Nx)] = 0.0;
    }
}

// Initialize u_prev = u^0, u_curr = u^1 via Taylor: u^1 = u^0 + dt*g + 0.5*(c dt)^2 Lap(u^0)
__global__ void kernel_init_u0_u1(double* __restrict__ u_prev,
                                  double* __restrict__ u_curr,
                                  int Nx, int Ny,
                                  double dx, double dy, double dt,
                                  double Lx, double Ly, double c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;

    const int id = IDX(i,j,Nx);
    const double x = i * dx;
    const double y = j * dy;

    // u^0
    double u0 = f0(x, y, Lx, Ly);
    u_prev[id] = u0;
    u_curr[id] = u0; // temp; interior overwritten below

    // boundaries handled by separate kernel
    if (i==0 || i==Nx-1 || j==0 || j==Ny-1) return;

    // Lap(u0)
    const double inv_dx2 = 1.0 / (dx*dx);
    const double inv_dy2 = 1.0 / (dy*dy);
    // Need neighbors from u_prev: but we just wrote u_prev[id] = u0.
    // To access neighbor u0 values, recompute f0 for neighbors (simple & safe baseline).
    double u0_ip1 = f0((i+1)*dx, y, Lx, Ly);
    double u0_im1 = f0((i-1)*dx, y, Lx, Ly);
    double u0_jp1 = f0(x, (j+1)*dy, Lx, Ly);
    double u0_jm1 = f0(x, (j-1)*dy, Lx, Ly);
    double lap = (u0_ip1 - 2.0*u0 + u0_im1) * inv_dx2
               + (u0_jp1 - 2.0*u0 + u0_jm1) * inv_dy2;

    double v0 = g0(x,y);
    const double c2dt2 = (c*c) * (dt*dt);
    u_curr[id] = u0 + dt * v0 + 0.5 * c2dt2 * lap;
}

__global__ void kernel_leapfrog_step(double* __restrict__ u_next,
                                     const double* __restrict__ u_curr,
                                     const double* __restrict__ u_prev,
                                     int Nx, int Ny,
                                     double dx, double dy,
                                     double c, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= Nx-1 || j <= 0 || j >= Ny-1) return; // interior only

    const int id = IDX(i,j,Nx);
    const double inv_dx2 = 1.0 / (dx*dx);
    const double inv_dy2 = 1.0 / (dy*dy);
    const double c2dt2   = (c*c) * (dt*dt);

    double lap = (u_curr[IDX(i+1,j,Nx)] - 2.0*u_curr[id] + u_curr[IDX(i-1,j,Nx)]) * inv_dx2
               + (u_curr[IDX(i,j+1,Nx)] - 2.0*u_curr[id] + u_curr[IDX(i,j-1,Nx)]) * inv_dy2;

    u_next[id] = 2.0*u_curr[id] - u_prev[id] + c2dt2 * lap;
}

// ---------------- host utilities ----------------
static void write_profile_csv(const std::string& fname,
                              const std::vector<double>& u,
                              int Nx, int Ny,
                              double dx, double dy,
                              double t, double Lx, double Ly, double c) {
    std::ofstream fp(fname);
    if (!fp) { std::cerr << "Could not open " << fname << " for writing.\n"; exit(1); }
    fp << std::fixed << std::setprecision(8);
    fp << "x,y,u_ex,u_num\n";
    for (int j=0; j<Ny; ++j) {
        const double y = j * dy;
        for (int i=0; i<Nx; ++i) {
            const double x = i * dx;
            const double ue = exact_solution(x,y,t,Lx,Ly,c);
            fp << x << "," << y << "," << ue << "," << u[IDX(i,j,Nx)] << "\n";
        }
    }
}

static void compute_error_host(const std::vector<double>& u,
                               int Nx, int Ny,
                               double dx, double dy, double t,
                               double Lx, double Ly, double c,
                               double& L1, double& L2, double& relL2) {
    long long N = (long long)Nx * (long long)Ny;
    long double sum_l1=0.0L, sum_l2=0.0L, sum_ue2=0.0L;
    for (int j=0; j<Ny; ++j) {
        double y = j * dy;
        for (int i=0; i<Nx; ++i) {
            double x = i * dx;
            double ue = exact_solution(x,y,t,Lx,Ly,c);
            double diff = u[IDX(i,j,Nx)] - ue;
            sum_l1 += fabsl(diff) * dx * dy;
            sum_l2 += (long double)diff * diff * dx * dy;
            sum_ue2 += (long double)ue * ue * dx * dy;
        }
    }
    L1 = (double)sum_l1;
    L2 = std::sqrt((double)sum_l2);
    relL2 = (sum_ue2>0) ? std::sqrt((double)(sum_l2/sum_ue2)) : 0.0;
}

// One simulation for given grid & CFL; returns errors;
static float run_simulation_cuda(int Nx, int Ny, double CFL, double T,
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
    dt = T / steps;
    const double t_final = dt * steps;

    const size_t N = (size_t)Nx * (size_t)Ny;

    // device arrays
    double *d_prev=nullptr, *d_curr=nullptr, *d_next=nullptr;
    CUDA_CHECK(cudaMalloc(&d_prev, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_curr, N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_next, N*sizeof(double)));

    dim3 block(32,8);
    dim3 grid((Nx+block.x-1)/block.x, (Ny+block.y-1)/block.y);

    // init (u^0 and u^1)
    CUDA_CHECK(cudaMemset(d_prev, 0, N*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_curr, 0, N*sizeof(double)));

    kernel_init_u0_u1<<<grid, block>>>(d_prev, d_curr, Nx, Ny, dx, dy, dt, Lx, Ly, c);
    CUDA_CHECK(cudaGetLastError());
    kernel_set_zero_boundary<<<grid, block>>>(d_prev, Nx, Ny);
    CUDA_CHECK(cudaGetLastError());
    kernel_set_zero_boundary<<<grid, block>>>(d_curr, Nx, Ny);
    CUDA_CHECK(cudaGetLastError());

    // timing
    cudaEvent_t evStart, evStop; CUDA_CHECK(cudaEventCreate(&evStart)); CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));

    // time loop
    for (int step=1; step<steps; ++step) {
        kernel_leapfrog_step<<<grid, block>>>(d_next, d_curr, d_prev, Nx, Ny, dx, dy, c, dt);
        CUDA_CHECK(cudaGetLastError());
        kernel_set_zero_boundary<<<grid, block>>>(d_next, Nx, Ny);
        CUDA_CHECK(cudaGetLastError());
        // rotate pointers
        double* tmp = d_prev; d_prev = d_curr; d_curr = d_next; d_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms=0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    CUDA_CHECK(cudaEventDestroy(evStart)); CUDA_CHECK(cudaEventDestroy(evStop));

    // copy back final field
    std::vector<double> h_curr(N);
    CUDA_CHECK(cudaMemcpy(h_curr.data(), d_curr, N*sizeof(double), cudaMemcpyDeviceToHost));

    // compute error on host
    compute_error_host(h_curr, Nx, Ny, dx, dy, t_final, Lx, Ly, c, L1, L2, relL2);

    if (write_profile) {
        char fname[256];
        std::snprintf(fname, sizeof(fname), "profile2D_CFL%.2f_Nx%05d_Ny%05d.csv", CFL, Nx, Ny);
        write_profile_csv(fname, h_curr, Nx, Ny, dx, dy, t_final, Lx, Ly, c);
    }

    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_next));

    return ms; // GPU elapsed milliseconds (kernels only)
}

int main() {
    const double Lx=1.0, Ly=1.0; const double c=1.0; const double T=1.0; // match CPU
    const double CFLs[] = {0.20, 0.50, 0.90, 1.05};
    const int nCFL = (int)(sizeof(CFLs)/sizeof(CFLs[0]));

    // Grid list (square grids). Feel free to adjust.
    const int Nlist[][2] = { {129,129}, {257,257}, {513,513} };
    const int nN = (int)(sizeof(Nlist)/sizeof(Nlist[0]));

    for (int cidx=0; cidx<nCFL; ++cidx) {
        const double CFL = CFLs[cidx];
        char fname[128];
        std::snprintf(fname, sizeof(fname), "error_results_leapfrog2D_CFL%.2f.csv", CFL);
        std::ofstream out(fname);
        if (!out) { std::cerr << "Could not open " << fname << " for writing.\n"; return 1; }
        out << std::fixed << std::setprecision(8);
        out << "CFL,Nx,Ny,dx,dy,dt,l1_error,l2_error,relative_l2,order_p,CPU_time(s)\n";

        double prev_dx=-1.0, prev_L2=-1.0;

        for (int i=0; i<nN; ++i) {
            const int Nx = Nlist[i][0];
            const int Ny = Nlist[i][1];
            const double dx = Lx / (Nx - 1);
            const double dy = Ly / (Ny - 1);

            const bool write_profile = (i==nN-1);
            double L1=0.0, L2=0.0, relL2=0.0;
            float ms = run_simulation_cuda(Nx, Ny, CFL, T, Lx, Ly, c, L1, L2, relL2, write_profile);

            // recompute dt aligned with ceil logic for metadata printing
            const double inv_dx2 = 1.0/(dx*dx);
            const double inv_dy2 = 1.0/(dy*dy);
            const double dt_max  = 1.0 / ( c * std::sqrt(inv_dx2 + inv_dy2) );
            double dt = CFL * dt_max; int steps = (int)std::ceil(T/dt - 1e-12); if (steps<1) steps=1; dt = T/steps;

            double p = NAN;
            if (prev_L2>0.0 && prev_dx>0.0) {
                p = std::log(L2/prev_L2) / std::log(dx/prev_dx);
            }

            out << std::setprecision(2) << CFL << "," << Nx << "," << Ny << ","
                << std::setprecision(8) << dx << "," << dy << "," << dt << ","
                << std::scientific << L1 << "," << L2 << "," << relL2 << ",";
            if (std::isnan(p)) out << "NA,"; else out << std::fixed << std::setprecision(6) << p << ",";
            out << std::fixed << std::setprecision(6) << (ms/1000.0) << "\n"; // seconds

            std::cout << "GPU 2D CFL=" << CFL << " Nx=" << Nx << " Ny=" << Ny
                      << " L2=" << L2 << " relL2=" << relL2
                      << " time(s)=" << (ms/1000.0) << (std::isnan(p)?" p=NA":" p≈2") << "\n";

            prev_dx = dx; prev_L2 = L2;
        }
        out.close();
    }

    std::cout << "Done. One CSV per CFL (with observed order p). 2D GPU profiles written for largest grids." << std::endl;
    return 0;
}
