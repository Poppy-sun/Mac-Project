#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define V   1.0      // wave speed
#define T_VAL 10.23   // final simulation time (superconvergence at T=1 ,T=10)

// Analytical Solution
double exact_solution(double x, double t) {
    return sin(M_PI * x) * cos(M_PI * t);
}

static inline double g0(double x) { return 0.0; }

// Initialization: 1) Dirichlet boundary 0; 2)initial speed,updated 3)first step
void initialize(double *u0, double *u1, int N, double dx, double dt) {
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        u0[i] = sin(M_PI * x);  // u(x, 0)
    }
    double r  = V * dt / dx;
    double r2 = r * r;

    // Dirichlet
    u0[0] = u0[N - 1] = 0.0;
    u1[0] = u1[N - 1] = 0.0;

    for (int j = 1; j <= N - 2; ++j) {
        double x   = j * dx;
        (void)x; 
        double lap = (u0[j+1] - 2.0 * u0[j] + u0[j-1]); // Δxx * dx^2 的“分子”
        double gj  = g0(x);
        // u^1 = u^0 + dt * g + 0.5 * (V*dt/dx)^2 * lap
        u1[j] = u0[j] + dt * gj + 0.5 * r2 * lap;
    }
}

//  Leapfrog implementation
void leapfrog(double *u_new, double *u_now, double *u_old, int N, double dx, double dt) {
    double coeff = (V * dt / dx) * (V * dt / dx);
    for (int i = 1; i < N - 1; i++) {
        u_new[i] = 2.0 * u_now[i] - u_old[i] + coeff * (u_now[i + 1] - 2.0 * u_now[i] + u_now[i - 1]);
    }
    u_new[0] = u_new[N - 1] = 0.0; // Dirichlet
}

// Errors measurement
void compute_error(double *u_num, int N, double dx, double t,
                   double *l1, double *l2, double *rel_l2) {
    double sum_l1 = 0.0, sum_l2 = 0.0, sum_exact2 = 0.0;
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        double u_exact = exact_solution(x, t);
        double diff = u_num[i] - u_exact;
        sum_l1    += fabs(diff) * dx;      // ~ \int |e| dx
        sum_l2    += diff * diff * dx;     // ~ \int e^2 dx
        sum_exact2 += u_exact * u_exact * dx;
    }
    *l1 = sum_l1;
    *l2 = sqrt(sum_l2);
    *rel_l2 = (sum_exact2 > 0.0) ? sqrt(sum_l2 / sum_exact2) : 0.0;
}

// Run a simulation and return the error (Return error & end of alignment & writable profile)
void run_simulation(int N, double CFL_val, double T_val,
                    double *out_l1, double *out_l2, double *out_rel_l2,
                    int write_profile) {
    double dx = 1.0 / (N - 1);
    double dt = CFL_val * dx / V;

    
    int steps = (int)ceil(T_val / dt - 1e-12);
    if (steps < 1) steps = 1;
    dt = T_val / steps;          
    double t_final = dt * steps; 

    // allocation
    double *u_old = (double*)malloc(N * sizeof(double));
    double *u_now = (double*)malloc(N * sizeof(double));
    double *u_new = (double*)malloc(N * sizeof(double));
    if (!u_old || !u_now || !u_new) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    //initialize
    initialize(u_old, u_now, N, dx, dt);

   //Advance time
    for (int step = 1; step < steps; step++) {
        leapfrog(u_new, u_now, u_old, N, dx, dt);
        double *tmp = u_old; u_old = u_now; u_now = u_new; u_new = tmp;
    }

    //get errors
    double l1, l2, rel_l2;
    compute_error(u_now, N, dx, t_final, &l1, &l2, &rel_l2);
    if (out_l1)     *out_l1 = l1;
    if (out_l2)     *out_l2 = l2;
    if (out_rel_l2) *out_rel_l2 = rel_l2;

    // profile
    if (write_profile) {
        char filename[128];
        sprintf(filename, "profile_CFL%.2f_N%05d.csv", CFL_val, N);
        FILE *fp_profile = fopen(filename, "w");
        if (!fp_profile) {
            fprintf(stderr, "Could not open %s for writing.\n", filename);
            exit(1);
        }
        fprintf(fp_profile, "x,u_ex,u_num\n");
        for (int i = 0; i < N; i++) {
            double x = i * dx;
            double u_ex = exact_solution(x, t_final);
            fprintf(fp_profile, "%.8f,%.8f,%.8f\n", x, u_ex, u_now[i]);
        }
        fclose(fp_profile);
    }

    free(u_old); free(u_now); free(u_new);
}

int main() {
   
    const double CFLs[] = {0.20, 0.50, 0.90, 1.05}; //the last one for display unstability
    const int nCFL = (int)(sizeof(CFLs) / sizeof(CFLs[0]));

    for (int c = 0; c < nCFL; ++c) {
        double CFL_val = CFLs[c];

        char fname[128];
        sprintf(fname, "error_results_leapfrog_CFL%.2f.csv", CFL_val);
        FILE *fp_out = fopen(fname, "w");
        if (!fp_out) {
            fprintf(stderr, "Could not open %s for writing.\n", fname);
            return 1;
        }
        fprintf(fp_out, "CFL,N,dx,dt,l1_error,l2_error,relative_l2,order_p,CPU_time(s)\n");

        double prev_dx = -1.0, prev_l2 = -1.0;

        int Ns[]={256,512,1024,2048};
        int numNs=sizeof(Ns)/sizeof(Ns[0]);

        for (int i=0; i<numNs; i++){     
            int N =Ns[i];
            double l1=0.0, l2=0.0, rel_l2=0.0;
            int write_profile = (i == numNs - 1);

            printf("Running CFL=%.2f, N=%d\n", CFL_val, N);
            // start time
            double start_time = omp_get_wtime();
            run_simulation(N, CFL_val, T_VAL, &l1, &l2, &rel_l2, write_profile);
            double end_time = omp_get_wtime();
            double elapsed = end_time - start_time;
            // end time

            double dx = 1.0 / (N - 1);
            double dt = CFL_val * dx / V;
            int steps = (int)ceil(T_val / dt - 1e-12);
            if (steps < 1) steps = 1;
            dt = T_VAL / steps;

            // calculate the convergence order
            double p = NAN;
            if (prev_l2 > 0.0) {
                // p = log(E(h)/E(h/2)) / log(2)
                p = log(l2 / prev_l2) / log( (dx) / (prev_dx) );
            }
            fprintf(fp_out, "%.2f,%d,%.8f,%.8f,%.8e,%.8e,%.8e,",
                    CFL_val, N, dx, dt, l1, l2, rel_l2);
            if (isnan(p)) fprintf(fp_out, "NA,%.6f\n",elapsed);
            else          fprintf(fp_out, "%.6f,%.6f\n", p, elapsed);

            prev_dx = dx;
            prev_l2 = l2;
        }
        fclose(fp_out);
    }

    printf("Done. One CSV per CFL (with observed order p). Profiles written for largest N.\n");
    return 0;
}
