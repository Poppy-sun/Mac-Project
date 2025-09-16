function make_perf_figs()
% make_perf_figs.m (v2)
% 读取 CPU 与 GPU 的计时 CSV，画：
% 1) Total runtime vs N (log-log)
% 2) Speedup (CPU / GPU total) vs N
% 3) GPU breakdown at a chosen N (H2D/Kernel/D2H)
%
CFL_tag = '0.50';   
T_label = '10.23';  

cpu_file = sprintf('error_results_leapfrog_CFL%s.csv', CFL_tag);
gpu_file = 'timings.csv';

assert(isfile(cpu_file),  "CPU file not found: %s", cpu_file);
assert(isfile(gpu_file),  "GPU file not found: %s", gpu_file);

CPU = readtable(cpu_file);
GPU = readtable(gpu_file);


CPU.Properties.VariableNames = matlab.lang.makeValidName(CPU.Properties.VariableNames);
GPU.Properties.VariableNames = matlab.lang.makeValidName(GPU.Properties.VariableNames);


cpuTimeCol = '';
cands = ["CPU_time_s_","CPU_time_s","CPU_time","CPU_times","CPU_times_"];
for k=1:numel(cands)
    if any(strcmpi(CPU.Properties.VariableNames, cands(k)))
        cpuTimeCol = cands(k); break;
    end
end
if isempty(cpuTimeCol)
    error("没找到 CPU_time(s) 列，请确认 CPU CSV 末列包含 CPU_time(s)。");
end

% --- 只用两边共同的 N（保证对齐）---
N_cpu = CPU.N(:);
N_gpu = GPU.N(:);
N = intersect(N_cpu, N_gpu);
if isempty(N), error('CPU 与 GPU 的 N 没有交集。'); end

CPU = sortrows(CPU(ismember(CPU.N,N),:),'N');
GPU = sortrows(GPU(ismember(GPU.N,N),:),'N');

% --- 数组化 ---
N           = CPU.N;
t_cpu       = CPU.(cpuTimeCol);        % 秒
t_gpu_total = GPU.total_ms/1000;       % 秒
t_h2d       = GPU.h2d_ms/1000;
t_kernel    = GPU.kernel_ms/1000;
t_d2h       = GPU.d2h_ms/1000;

% --- 基本 sanity check ---
if any(t_cpu<=0) || any(t_gpu_total<=0)
    warning('检测到非正时间值，请检查原始 CSV。');
end

% 加速比 
speedup = t_cpu ./ t_gpu_total;

% ========= 1) Total runtime vs N =========
figure('Color','w');
loglog(N, t_cpu, 'o-','LineWidth',1.6); hold on; grid on;
loglog(N, t_gpu_total, 's-','LineWidth',1.6);
xlabel('N (grid size)'); ylabel('Time (s)');
title(sprintf('Total runtime vs N (CFL=%s, T=%s)', CFL_tag, T_label));
legend('CPU total','GPU total','Location','northwest');
saveas(gcf, 'perf_time_vs_N.png');
exportgraphics(gcf, 'perf_time_vs_N.pdf','ContentType','vector');

% ========= 2) Speedup vs N =========
figure('Color','w');
plot(N, speedup, 'o-','LineWidth',1.6); grid on;
xlabel('N (grid size)'); ylabel('Speedup (CPU / GPU total)');
title(sprintf('Speedup vs N (CFL=%s, T=%s)', CFL_tag, T_label));
yline(1,'--','No speedup','LabelHorizontalAlignment','left');
saveas(gcf, 'perf_speedup_vs_N.png');
exportgraphics(gcf, 'perf_speedup_vs_N.pdf','ContentType','vector');

% ========= 3) GPU breakdown at max N =========
[~,idxMax] = max(N);
figure('Color','w');
bar([t_h2d(idxMax), t_kernel(idxMax), t_d2h(idxMax)]);
set(gca,'XTickLabel',{'H2D','Kernel','D2H'});
ylabel('Time (s)');
title(sprintf('GPU breakdown at N=%d', N(idxMax)));
grid on;
saveas(gcf, 'perf_gpu_breakdown_maxN.png');
exportgraphics(gcf, 'perf_gpu_breakdown_maxN.pdf','ContentType','vector');


T = table(N, t_cpu, t_gpu_total, speedup, ...
          'VariableNames', {'N','CPU_s','GPU_total_s','Speedup'});
disp('==== Summary (aligned by N) ====');
disp(T);

fprintf('Saved: perf_time_vs_N.[png|pdf], perf_speedup_vs_N.[png|pdf], perf_gpu_breakdown_maxN.[png|pdf]\n');
end