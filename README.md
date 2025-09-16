This repository contains the implementation and analysis of wave equation solvers with progressive optimization using CUDA parallel computing. The project demonstrates the performance improvements achieved through GPU acceleration and optimization techniques for 1D and 2D wave simulations.
Project Overview

This research project focuses on:
1.Explicit finite difference methods for solving 1D and 2D wave equations
2.Leapfrog scheme implementation with second-order accuracy in space and time
3.CFL stability analysis and convergence verification
4.GPU acceleration using CUDA with progressive optimization techniques
5.Performance benchmarking comparing CPU vs GPU implementations

📁 Repository Structure
├── 1D_Code/                    # 1D wave equation implementations
│   ├── CUDA_Baseline.cu        # Basic CUDA implementation
│   ├── CUDA_Optimized.cu       # Optimized CUDA version
│   ├── C_Code.c                # CPU reference implementation
│   ├── accuracy_test.m         # MATLAB accuracy verification
│   ├── make_perf_figs.m        # Performance visualization
│   └── time.m                  # Timing analysis scripts
├── 2D_Code/                    # 2D wave equation implementations
│   ├── CUDA_Baseline.cu        # Basic CUDA implementation
│   ├── CUDA_Optimized.cu       # Optimized CUDA version
│   ├── C_Code.c                # CPU reference implementation
│   ├── accuracy_test.m         # MATLAB accuracy verification
│   ├── make_perf_figs.m        # Performance visualization
│   └── time.m                  # Timing analysis scripts
├── Dissertation_LishuangSun.pdf # Complete dissertation
└── Final Project Presentation.pdf # Project presentation slides

Mathematical Foundation
1D Wave Equation

Governing equation: ∂²u/∂t² = c²∂²u/∂x²
Domain: x ∈ [0,1] with Dirichlet boundary conditions
Test case: u(x,0) = sin(πx), ∂u/∂t(x,0) = 0
Analytical solution: u(x,t) = sin(πx)cos(πt)

2D Wave Equation

Governing equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
Discretization: 5-point spatial stencil
Stability condition: λ²ₓ + λ²ᵧ ≤ 1 (2D CFL condition)
Test case: Standing wave with separable initial conditions

Implementation Strategy
1. CPU Implementation (Baseline)
Pure C implementation of finite difference methods
Serves as accuracy and performance reference
Standard numerical schemes for wave equation discretization

2. CUDA Baseline Implementation
Direct GPU port of CPU algorithms
Basic thread parallelization
Performance comparison with CPU version

3. CUDA Optimized Implementation
Advanced GPU memory management
Shared memory optimization
Thread block optimization
Coalesced memory access patterns

Key Results
Numerical Accuracy
Convergence rate: Confirmed second-order accuracy O(h²)
Superconvergence: Observed at integer times for smooth initial data
Stability verification: CFL condition λ ≤ 1 (1D), λ²ₓ + λ²ᵧ ≤ 1 (2D)

Performance Improvements
1D Results: Up to 23× speedup over CPU (optimized vs baseline: 18×)
2D Results: Up to 29× speedup for large grids (513×513)
Scalability: Performance improvements increase with problem size
Memory efficiency: Optimized kernels achieve near-bandwidth saturation

Optimization Impact
Baseline CUDA: Memory-bandwidth limited, significant transfer overhead
Optimized CUDA: Computation-dominated, reduced global memory traffic
Shared memory: Effective data reuse within thread blocks
Kernel fusion: Eliminated separate boundary handling passes

Technical Specifications
Hardware: NVIDIA GeForce RTX 2080, Intel Core i5-3570K
Software: CUDA 12.9, GCC compiler, MATLAB for analysis
Precision: Double precision for numerical accuracy
Grid sizes: Tested from 64 to 32,768 points (1D), up to 513×513 (2D)
