function make_perf_figs_2d()
% make_perf_figs_2d.m
% 读取 2D CPU 与 GPU 计时，绘制：
% 1) Total runtime vs Nx (log-log)
% 2) Speedup (CPU / GPU total) vs Nx
% 3) GPU breakdown at max Nx（若有 H2D/Kernel/D2H，否则给出 Kernel-only 提示）

clc; close all;

%% 
CFL_tag   = '0.50';   
T_label   = '1.00';   %

cpu_file  = sprintf('C_error_results_leapfrog2D_CFL%s.csv', CFL_tag);   % 来自 2D CPU C 程序
gpu_timeA = 'timings2D.csv';                                          % 推荐：2D CUDA 计时拆分
gpu_timeB = sprintf('G_error_results_leapfrog2D_CFL%s.csv', CFL_tag);   % 备选：2D CUDA 结果文件（最后列=GPU时间）

assert(isfile(cpu_file),  "CPU file not found: %s", cpu_file);

CPU = readtable(cpu_file);
CPU.Properties.VariableNames = matlab.lang.makeValidName(CPU.Properties.VariableNames);

% 找 CPU 时间列（兼容不同命名）
cpuTimeCol = find_col(CPU, ["CPU_time_s_","CPU_time_s__1_","CPU_time_s","CPU_time","CPU_times"]);
if isempty(cpuTimeCol)
    error("CPU CSV 未找到 CPU 时间列（末列通常命名为 'CPU_time(s)'）");
end

%% === 2) 读取 GPU 时间（优先 A，其次 B） ===
GPUhasBreakdown = false;
if isfile(gpu_timeA)
    GPU = readtable(gpu_timeA);
    GPU.Properties.VariableNames = matlab.lang.makeValidName(GPU.Properties.VariableNames);
    req = ["Nx","Ny","h2d_ms","kernel_ms","d2h_ms","total_ms"];
    assert(all(ismember(req, string(GPU.Properties.VariableNames))), ...
        'timings2D.csv 应包含列: %s', strjoin(req,', '));
    GPUhasBreakdown = true;
else
    assert(isfile(gpu_timeB), "找不到 GPU 计时：既没有 %s 也没有 %s", gpu_timeA, gpu_timeB);
    GPU = readtable(gpu_timeB);
    GPU.Properties.VariableNames = matlab.lang.makeValidName(GPU.Properties.VariableNames);
    % 用同名文件的“时间列”作为 GPU 总时间（kernel-only）
    gpuTimeCol = find_col(GPU, ["CPU_time_s_","CPU_time_s__1_","CPU_time_s","CPU_time","CPU_times"]);
    assert(~isempty(gpuTimeCol), "GPU 结果文件中未找到时间列。");
    GPU.total_ms = GPU.(gpuTimeCol) * 1000.0;   % s -> ms
    % 没有分解项时设为 NaN
    GPU.h2d_ms   = NaN(height(GPU),1);
    GPU.kernel_ms= GPU.total_ms;
    GPU.d2h_ms   = NaN(height(GPU),1);
end

%% === 3) 只用 Nx,Ny 交集对齐（二维要对齐两个维度）===
Nx_cpu = CPU.Nx(:);   Ny_cpu = CPU.Ny(:);
Nx_gpu = GPU.Nx(:);   Ny_gpu = GPU.Ny(:);

% 对齐 (Nx,Ny) 成对匹配
[Ng, ia, ib] = intersect([Nx_cpu Ny_cpu], [Nx_gpu Ny_gpu], 'rows');
if isempty(Ng)
    error('CPU 与 GPU 的 (Nx,Ny) 没有交集，请确保两边跑了相同网格。');
end
CPU = CPU(ia,:); GPU = GPU(ib,:);

% 排序（按 Nx 再 Ny）
[~, order] = sortrows(Ng, [1 2]); 
CPU = CPU(order,:); GPU = GPU(order,:);
Nx = CPU.Nx; Ny = CPU.Ny;

% 数组化时间
t_cpu       = CPU.(cpuTimeCol);     % 秒
t_gpu_total = GPU.total_ms/1000;    % 秒
t_h2d       = GPU.h2d_ms/1000;
t_kernel    = GPU.kernel_ms/1000;
t_d2h       = GPU.d2h_ms/1000;

% 基本检查
if any(t_cpu<=0) || any(t_gpu_total<=0)
    warning('检测到非正时间值，请核对原始 CSV。');
end

% 加速比
speedup = t_cpu ./ t_gpu_total;

%% === 4) Total runtime vs Nx（loglog，用 Nx 表示单边网格尺度） ===
figure('Color','w');
loglog(Nx, t_cpu, 'o-','LineWidth',1.6); hold on; grid on;
loglog(Nx, t_gpu_total, 's-','LineWidth',1.6);
xlabel('N_x (grid per side)'); ylabel('Time (s)');
title(sprintf('2D Total runtime vs N_x (CFL=%s, T=%s)', CFL_tag, T_label));
legend('CPU total','GPU total','Location','northwest');
saveas(gcf, '2d_perf_time_vs_Nx.png');
exportgraphics(gcf, '2d_perf_time_vs_Nx.pdf','ContentType','vector');

%% === 5) Speedup vs Nx ===
figure('Color','w');
plot(Nx, speedup, 'o-','LineWidth',1.6); grid on;
xlabel('N_x (grid per side)'); ylabel('Speedup (CPU / GPU total)');
title(sprintf('2D Speedup vs N_x (CFL=%s, T=%s)', CFL_tag, T_label));
yline(1,'--','No speedup','LabelHorizontalAlignment','left');
saveas(gcf, '2d_perf_speedup_vs_Nx.png');
exportgraphics(gcf, '2d_perf_speedup_vs_Nx.pdf','ContentType','vector');

%% === 6) GPU breakdown at max Nx（如果有分解） ===
[~, idxMax] = max(Nx);
figure('Color','w');
if GPUhasBreakdown
    bar([t_h2d(idxMax), t_kernel(idxMax), t_d2h(idxMax)]);
    set(gca,'XTickLabel',{'H2D','Kernel','D2H'});
    ylabel('Time (s)');
    title(sprintf('2D GPU breakdown at Nx=%d, Ny=%d', Nx(idxMax), Ny(idxMax)));
else
    bar([t_kernel(idxMax)]);
    set(gca,'XTickLabel',{'Kernel'});
    ylabel('Time (s)');
    title(sprintf('2D GPU (kernel-only) at Nx=%d, Ny=%d', Nx(idxMax), Ny(idxMax)));
end
grid on;
saveas(gcf, '2d_perf_gpu_breakdown_maxN.png');
exportgraphics(gcf, '2d_perf_gpu_breakdown_maxN.pdf','ContentType','vector');

%%
T = table(Nx, Ny, t_cpu, t_gpu_total, speedup, ...
          'VariableNames', {'Nx','Ny','CPU_s','GPU_total_s','Speedup'});
disp('==== 2D Summary (aligned by Nx,Ny) ====');
disp(T);

fprintf('Saved: 2d_perf_time_vs_Nx.[png|pdf], 2d_perf_speedup_vs_Nx.[png|pdf], 2d_perf_gpu_breakdown_maxN.[png|pdf]\n');

end

% ---------- helpers ----------
function name = find_col(T, candidates)
    name = '';
    for k = 1:numel(candidates)
        if any(strcmpi(T.Properties.VariableNames, candidates(k)))
            name = candidates(k); return;
        end
    end
end
