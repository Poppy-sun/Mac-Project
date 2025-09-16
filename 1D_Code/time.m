
clc; close all;

%% read tables
CFL_tag = '0.50';                         
cpu_file = sprintf('error_results_leapfrog_CFL%s.csv', CFL_tag);
gpu_file = 'timings.csv';

assert(isfile(cpu_file),  "CPU file not found: %s", cpu_file);
assert(isfile(gpu_file),  "GPU file not found: %s", gpu_file);

CPU = readtable(cpu_file);                
GPU = readtable(gpu_file);                


CPU.Properties.VariableNames = matlab.lang.makeValidName(CPU.Properties.VariableNames);
GPU.Properties.VariableNames = matlab.lang.makeValidName(GPU.Properties.VariableNames);

% 
cpuTimeCol = '';
cand = ["CPU_time_s_", "CPU_time_s", "CPU_time_s__1_","CPU_time_s__","CPU_time"];
for k = 1:numel(cand)
    if any(strcmpi(CPU.Properties.VariableNames, cand(k)))
        cpuTimeCol = cand(k);
        break;
    end
end
if isempty(cpuTimeCol)
    error("CPU CSV 中未找到 'CPU_time(s)' 列。请确认你已按之前版本输出了 CPU_time(s)。");
end

% 提取并对齐 N（两侧交集）
N_cpu = CPU.N(:);
N_gpu = GPU.N(:);
N = intersect(N_cpu, N_gpu);
if isempty(N)
    error('CPU 与 GPU 的 N 集合没有交集，请确认两侧跑了相同的 N。');
end

% 按交集 N 取子表并排序
CPU_sub = sortrows(CPU(ismember(CPU.N, N), :), 'N');
GPU_sub = sortrows(GPU(ismember(GPU.N, N), :), 'N');

% 数组化
N = CPU_sub.N;                                  
t_cpu = CPU_sub.(cpuTimeCol);                   % 秒
t_gpu_total = GPU_sub.total_ms / 1000.0;         
t_h2d = GPU_sub.h2d_ms / 1000.0;                 
t_kernel = GPU_sub.kernel_ms / 1000.0;           
t_d2h = GPU_sub.d2h_ms / 1000.0;                

%
if any(t_gpu_total <= 0) || any(t_cpu <= 0)
    warning('检测到非正时间值，请核对原始 CSV。');
end

%%  calculate speedup
speedup = t_cpu ./ t_gpu_total;                 
%%  total time vs N ===
figure('Color','w');
loglog(N, t_cpu, 'o-','LineWidth',1.6); hold on; grid on;
loglog(N, t_gpu_total, 's-','LineWidth',1.6);
xlabel('N (grid size)'); ylabel('Time (s)');
title(sprintf('Total runtime vs N (CFL=%s, T=%.2f)', CFL_tag, readTfromCPU(CPU)));
legend('CPU total','GPU total','Location','northwest');
saveas(gcf, 'perf_time_vs_N.png');
exportgraphics(gcf, 'perf_time_vs_N.pdf', 'ContentType','vector');

%%\ speedup vs N ===
figure('Color','w');
plot(N, speedup, 'o-','LineWidth',1.6); grid on;
xlabel('N (grid size)'); ylabel('Speedup (CPU time / GPU total)');
title(sprintf('Speedup vs N (CFL=%s, T=%.2f)', CFL_tag, readTfromCPU(CPU)));
yline(1,'--','No speedup'); % 参考线
saveas(gcf, 'perf_speedup_vs_N.png');
exportgraphics(gcf, 'perf_speedup_vs_N.pdf', 'ContentType','vector');


[~, idxMax] = max(N);
figure('Color','w');
bar([t_h2d(idxMax), t_kernel(idxMax), t_d2h(idxMax)]);
set(gca,'XTickLabel',{'H2D','Kernel','D2H'});
ylabel('Time (s)'); title(sprintf('GPU breakdown at N=%d', N(idxMax)));
grid on;
saveas(gcf, 'perf_gpu_breakdown_maxN.png');
exportgraphics(gcf, 'perf_gpu_breakdown_maxN.pdf', 'ContentType','vector');

T = table(N, t_cpu, t_gpu_total, speedup, ...
    'VariableNames', {'N','CPU_s','GPU_total_s','Speedup'});
disp('==== Summary (aligned N only) ====');
disp(T);

fprintf('Saved: perf_time_vs_N.[png|pdf], perf_speedup_vs_N.[png|pdf], perf_gpu_breakdown_maxN.[png|pdf]\n');

end

function Tval = readTfromCPU(CPU)
try
    if any(strcmpi(CPU.Properties.VariableNames,'dt'))
        dt = CPU.dt(1);
        
        Tval = NaN;
    else
        Tval = NaN;
    end
catch
    Tval = NaN;
end
if isnan(Tval), Tval = 1.00; end    
end