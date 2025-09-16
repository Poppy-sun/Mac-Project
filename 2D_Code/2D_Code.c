
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/*
  2D wave equation with Dirichlet boundaries solved by Leapfrog,
  analytical error evaluation and CSV outputs.

    u_tt = c^2 (u_xx + u_yy)  on (0,Lx) x (0,Ly)
    
  Standing-wave analytical solution used for error checks:
    u(x,y,t) = sin(pi*x/Lx) * sin(pi*y/Ly) * cos(omega * t),
    omega = c*pi*sqrt(1/Lx^2 + 1/Ly^2).

  First step from Taylor expansion (g=0 by default here):
    u^1 = u^0 + dt*g + 0.5*(c*dt)^2*Lap(u^0)
*/

#define IDX(i,j,Nx) ((i) + (Nx)*(j))

static inline double exact_solution(double x, double y, double t,
                                    double Lx, double Ly, double c) {
    const double kx = M_PI / Lx;
    const double ky = M_PI / Ly;
    const double omega = c * M_PI * sqrt(1.0/(Lx*Lx) + 1.0/(Ly*Ly));
    return sin(kx * x) * sin(ky * y) * cos(omega * t);
}

static inline double f0(double x, double y, double Lx, double Ly) {
    const double kx = M_PI / Lx;
    const double ky = M_PI / Ly;
    return sin(kx * x) * sin(ky * y);
}

static inline double g0(double x, double y) {
    (void)x; (void)y;
    return 0.0;
}

static void compute_error_2d(const double *u, int Nx, int Ny,
                             double dx, double dy, double t,
                             double Lx, double Ly, double c,
                             double *l1, double *l2, double *rel_l2) {
    double sum_l1 = 0.0, sum_l2 = 0.0, sum_ue2 = 0.0;
    for (int j = 0; j < Ny; ++j) {
        const double y = j * dy;
        for (int i = 0; i < Nx; ++i) {
            const double x = i * dx;
            const double ue = exact_solution(x, y, t, Lx, Ly, c);
            const double diff = u[IDX(i,j,Nx)] - ue;
            sum_l1 += fabs(diff) * dx * dy;
            sum_l2 += diff * diff * dx * dy;
            sum_ue2 += ue * ue * dx * dy;
        }
    }
    *l1 = sum_l1;
    *l2 = sqrt(sum_l2);
    *rel_l2 = (sum_ue2 > 0.0) ? sqrt(sum_l2 / sum_ue2) : 0.0;
}

static void write_profile_csv(const char *fname,
                              const double *u, int Nx, int Ny,
                              double dx, double dy,
                              double t, double Lx, double Ly, double c) {
    FILE *fp = fopen(fname, "w");
    if (!fp) { fprintf(stderr, "Could not open %s for writing.\n", fname); exit(1); }
    fprintf(fp, "x,y,u_ex,u_num\n");
    for (int j = 0; j < Ny; ++j) {
        const double y = j * dy;
        for (int i = 0; i < Nx; ++i) {
            const double x = i * dx;
            const double ue = exact_solution(x, y, t, Lx, Ly, c);
            fprintf(fp, "%.8f,%.8f,%.8f,%.8f\n", x, y, ue, u[IDX(i,j,Nx)]);
        }
    }
    fclose(fp);
}

static void initialize(double *u_prev, double *u_curr,
                       int Nx, int Ny, double dx, double dy,
                       double dt, double Lx, double Ly, double c) {
    const double inv_dx2 = 1.0 / (dx*dx);
    const double inv_dy2 = 1.0 / (dy*dy);
    const double c2dt2   = (c*c) * (dt*dt);

    for (int j = 0; j < Ny; ++j) {
        const double y = j * dy;
        for (int i = 0; i < Nx; ++i) {
            const double x = i * dx;
            u_prev[IDX(i,j,Nx)] = f0(x, y, Lx, Ly);
            u_curr[IDX(i,j,Nx)] = u_prev[IDX(i,j,Nx)];
        }
    }

    for (int i = 0; i < Nx; ++i) { u_prev[IDX(i,0,Nx)] = 0.0; u_prev[IDX(i,Ny-1,Nx)] = 0.0; }
    for (int j = 0; j < Ny; ++j) { u_prev[IDX(0,j,Nx)] = 0.0; u_prev[IDX(Nx-1,j,Nx)] = 0.0; }

    for (int j = 1; j < Ny-1; ++j) {
        for (int i = 1; i < Nx-1; ++i) {
            const int id = IDX(i,j,Nx);
            const double lap = (u_prev[IDX(i+1,j,Nx)] - 2.0*u_prev[id] + u_prev[IDX(i-1,j,Nx)]) * inv_dx2
                             + (u_prev[IDX(i,j+1,Nx)] - 2.0*u_prev[id] + u_prev[IDX(i,j-1,Nx)]) * inv_dy2;
            const double v0 = g0(i*dx, j*dy);
            u_curr[id] = u_prev[id] + dt * v0 + 0.5 * c2dt2 * lap;
        }
    }

    for (int i = 0; i < Nx; ++i) { u_curr[IDX(i,0,Nx)] = 0.0; u_curr[IDX(i,Ny-1,Nx)] = 0.0; }
    for (int j = 0; j < Ny; ++j) { u_curr[IDX(0,j,Nx)] = 0.0; u_curr[IDX(Nx-1,j,Nx)] = 0.0; }
}

static void leapfrog_step(double *u_next, const double *u_curr, const double *u_prev,
                          int Nx, int Ny, double dx, double dy, double c, double dt) {
    const double inv_dx2 = 1.0 / (dx*dx);
    const double inv_dy2 = 1.0 / (dy*dy);
    const double c2dt2   = (c*c) * (dt*dt);

    for (int j = 1; j < Ny-1; ++j) {
        for (int i = 1; i < Nx-1; ++i) {
            const int id = IDX(i,j,Nx);
            const double lap = (u_curr[IDX(i+1,j,Nx)] - 2.0*u_curr[id] + u_curr[IDX(i-1,j,Nx)]) * inv_dx2
                             + (u_curr[IDX(i,j+1,Nx)] - 2.0*u_curr[id] + u_curr[IDX(i,j-1,Nx)]) * inv_dy2;
            u_next[id] = 2.0*u_curr[id] - u_prev[id] + c2dt2 * lap;
        }
    }

    for (int i = 0; i < Nx; ++i) { u_next[IDX(i,0,Nx)] = 0.0; u_next[IDX(i,Ny-1,Nx)] = 0.0; }
    for (int j = 0; j < Ny; ++j) { u_next[IDX(0,j,Nx)] = 0.0; u_next[IDX(Nx-1,j,Nx)] = 0.0; }
}

static void run_simulation_2d(int Nx, int Ny, double CFL, double T_val,
                              double Lx, double Ly, double c,
                              double *out_l1, double *out_l2, double *out_rel_l2,
                              int write_profile) {
    const double dx = Lx / (Nx - 1);
    const double dy = Ly / (Ny - 1);

    const double inv_dx2 = 1.0 / (dx*dx);
    const double inv_dy2 = 1.0 / (dy*dy);
    const double dt_max = 1.0 / ( c * sqrt(inv_dx2 + inv_dy2) );
    double dt = CFL * dt_max;

    int steps = (int)ceil(T_val / dt - 1e-12);
    if (steps < 1) steps = 1;
    dt = T_val / steps;
    const double t_final = dt * steps;

    const int N = Nx * Ny;
    double *u_prev = (double*)malloc((size_t)N * sizeof(double));
    double *u_curr = (double*)malloc((size_t)N * sizeof(double));
    double *u_next = (double*)malloc((size_t)N * sizeof(double));
    if (!u_prev || !u_curr || !u_next) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    initialize(u_prev, u_curr, Nx, Ny, dx, dy, dt, Lx, Ly, c);

    double t = dt;
    for (int step = 1; step < steps; ++step) {
        leapfrog_step(u_next, u_curr, u_prev, Nx, Ny, dx, dy, c, dt);
        double *tmp = u_prev; u_prev = u_curr; u_curr = u_next; u_next = tmp;
        t += dt;
    }

    double l1=0.0, l2=0.0, rel_l2=0.0;
    compute_error_2d(u_curr, Nx, Ny, dx, dy, t_final, Lx, Ly, c, &l1, &l2, &rel_l2);
    if (out_l1) *out_l1 = l1;
    if (out_l2) *out_l2 = l2;
    if (out_rel_l2) *out_rel_l2 = rel_l2;

    if (write_profile) {
        char fname[256];
        snprintf(fname, sizeof(fname), "profile2D_CFL%.2f_Nx%05d_Ny%05d.csv", CFL, Nx, Ny);
        write_profile_csv(fname, u_curr, Nx, Ny, dx, dy, t_final, Lx, Ly, c);
    }

    free(u_prev); free(u_curr); free(u_next);
}

int main(void) {
    const double CFLs[] = {0.20, 0.50, 0.90, 1.05};
    const int nCFL = (int)(sizeof(CFLs)/sizeof(CFLs[0]));

    const int Nlist[][2] = { {129,129}, {257,257}, {513,513} };
    const int nN = (int)(sizeof(Nlist)/sizeof(Nlist[0]));

    const double Lx = 1.0, Ly = 1.0;
    const double c = 1.0;
    const double T_VAL = 1.0;

    for (int cidx = 0; cidx < nCFL; ++cidx) {
        const double CFL = CFLs[cidx];

        char fname[128];
        snprintf(fname, sizeof(fname), "error_results_leapfrog2D_CFL%.2f.csv", CFL);
        FILE *fp_out = fopen(fname, "w");
        if (!fp_out) { fprintf(stderr, "Could not open %s for writing.\n", fname); return 1; }
        fprintf(fp_out, "CFL,Nx,Ny,dx,dy,dt,l1_error,l2_error,relative_l2,order_p,CPU_time(s)\n");

        double prev_dx = -1.0, prev_l2 = -1.0;

        for (int i = 0; i < nN; ++i) {
            const int Nx = Nlist[i][0];
            const int Ny = Nlist[i][1];
            const double dx = Lx / (Nx - 1);
            const double dy = Ly / (Ny - 1);

            const double start_time = (double)clock() / CLOCKS_PER_SEC;
            double l1=0.0, l2=0.0, rel_l2=0.0;
            const int write_profile = (i == nN - 1);
            run_simulation_2d(Nx, Ny, CFL, T_VAL, Lx, Ly, c, &l1, &l2, &rel_l2, write_profile);
            const double end_time = (double)clock() / CLOCKS_PER_SEC;
            const double elapsed = end_time - start_time;

            const double inv_dx2 = 1.0 / (dx*dx);
            const double inv_dy2 = 1.0 / (dy*dy);
            const double dt_max = 1.0 / ( c * sqrt(inv_dx2 + inv_dy2) );
            double dt = CFL * dt_max;
            int steps = (int)ceil(T_VAL / dt - 1e-12);
            if (steps < 1) steps = 1;
            dt = T_VAL / steps;

            double p = NAN;
            if (prev_l2 > 0.0 && prev_dx > 0.0) {
                p = log(l2 / prev_l2) / log(dx / prev_dx);
            }

            fprintf(fp_out, "%.2f,%d,%d,%.8f,%.8f,%.8f,%.8e,%.8e,%.8e,",
                    CFL, Nx, Ny, dx, dy, dt, l1, l2, rel_l2);
            if (isnan(p)) fprintf(fp_out, "NA,%.6f\n", elapsed);
            else          fprintf(fp_out, "%.6f,%.6f\n", p, elapsed);

            prev_dx = dx;
            prev_l2 = l2;

            printf("Running 2D CFL=%.2f, Nx=%d, Ny=%d -> L2=%.3e, relL2=%.3e, p=%s\n",
                   CFL, Nx, Ny, l2, rel_l2, isnan(p)?"NA":"computed");
        }
        fclose(fp_out);
    }

    printf("Done. One CSV per CFL (with observed order p). 2D profiles written for largest grids.\n");
    return 0;
}
