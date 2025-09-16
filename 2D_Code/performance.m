function plot_2d_cuda_opt_results()

clc; close all;
%%
CFL_tag   = '0.50';           
NxNy_max  = [513 513];        
T_label   = '1.00';           
save_figs = true;             

% 文件名（GPU 优化版）
gpu_err_file = sprintf('error_results_leapfrog2D_CFL%s.csv', CFL_tag);
gpu_prof_file = sprintf('profile2D_CFL%s_Nx%05d_Ny%05d.csv', CFL_tag, NxNy_max(1), NxNy_max(2));

% 可选：CPU 结果（若存在则比较）
cpu_err_file = sprintf('C_error_results_leapfrog2D_CFL%s.csv', CFL_tag); % 若你的 CPU 文件名就是 gpu 的名字，也会自动识别

%%read tables
assert(isfile(gpu_err_file), "找不到 GPU 误差文件：%s", gpu_err_file);
G = readtable(gpu_err_file);
G.Properties.VariableNames = matlab.lang.makeValidName(G.Properties.VariableNames);

reqG = ["CFL","Nx","Ny","dx","dy","dt","l1_error","l2_error","relative_l2","order_p"];
assert(all(ismember(reqG, string(G.Properties.VariableNames))), 'GPU 误差文件缺少必要列。');

% 识别“时间列”：GPU 内核时间（秒）
gpuTimeCol = find_col(G, ["CPU_time_s_","CPU_time_s__1_","CPU_time_s","CPU_time","CPU_times"]);
assert(~isempty(gpuTimeCol), 'GPU CSV 未找到时间列（末列应为 GPU kernel 时间，沿用 CPU_time(s) 字段名）。');

NxG = G.Nx(:); NyG = G.Ny(:);
t_gpu = G.(gpuTimeCol)(:);      % s

%% ================= read GPU profile=================
assert(isfile(gpu_prof_file), "找不到 GPU profile 文件：%s", gpu_prof_file);
P = readtable(gpu_prof_file);
P.Properties.VariableNames = matlab.lang.makeValidName(P.Properties.VariableNames);

% 期望列：x,y,u_ex,u_num
needP = ["x","y","u_ex","u_num"];
assert(all(ismember(needP, string(P.Properties.VariableNames))), 'profile2D CSV 缺列（x,y,u_ex,u_num）。');

xv = unique(P.x); nx = numel(xv);
yv = unique(P.y); ny = numel(yv);
assert(nx*ny == height(P), 'profile2D CSV 行数与 Nx*Ny 不一致。');

Ue = reshape(P.u_ex , [nx, ny])';   % 行=y，列=x
Un = reshape(P.u_num, [nx, ny])';
Err = Un - Ue;
[X,Y] = meshgrid(xv, yv);

%% ================= 读取 CPU 误差/时间并对齐 =================
hasCPU = isfile(cpu_err_file);
C = table();
if hasCPU
    C = readtable(cpu_err_file);
    C.Properties.VariableNames = matlab.lang.makeValidName(C.Properties.VariableNames);
    if ~all(ismember(["Nx","Ny"], string(C.Properties.VariableNames)))
        warning('CPU CSV 未包含 Nx/Ny 列，无法对齐比较。将仅绘制 GPU 性能。');
        hasCPU = false;
    end
end

% CPU 时间列名
if hasCPU
    cpuTimeCol = find_col(C, ["CPU_time_s_","CPU_time_s__1_","CPU_time_s","CPU_time","CPU_times"]);
    if isempty(cpuTimeCol)
        warning("CPU CSV 未找到 CPU 时间列，speedup 将跳过。");
        hasCPU = false;
    end
end

% 对齐 (Nx,Ny)
if hasCPU
    [NG, ia, ib] = intersect([C.Nx(:) C.Ny(:)], [NxG NyG], 'rows');
    if isempty(NG)
        warning('CPU 与 GPU 的 (Nx,Ny) 没有交集，跳过 speedup 图。');
        hasCPU = false;
    else
        C = C(ia,:); G2 = G(ib,:);
        % 排序
        [~, ord] = sortrows(NG,[1 2]);
        C = C(ord,:); G2 = G2(ord,:);
        Nx_s = C.Nx; Ny_s = C.Ny;
        t_cpu = C.(cpuTimeCol);
        t_gpu_s = G2.(gpuTimeCol);
        speedup = t_cpu ./ t_gpu_s;
    end
end

%% ================= 1) exact vs analytical =================
figure('Color','w','Name','Numerical vs Analytical (surf)');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
surf(X,Y,Un,'EdgeColor','none'); view(35,35); axis tight; colormap turbo; colorbar
set(gca,'YDir','normal');
title(sprintf('Numerical u(x,y,T), CFL=%s, %dx%d',CFL_tag,nx,ny));
xlabel('x'); ylabel('y'); zlabel('u');

nexttile;
surf(X,Y,Ue,'EdgeColor','none'); view(35,35); axis tight; colormap turbo; colorbar
set(gca,'YDir','normal');
title('Analytical u_{exact}(x,y,T)');
xlabel('x'); ylabel('y'); zlabel('u');

if save_figs, save_pair('2d_num_vs_exact_surf'); end

%% ================= 2) error field (heatmap) =================
figure('Color','w','Name','Error heatmap');
imagesc(xv, yv, abs(Err)); axis image; set(gca,'YDir','normal');
colormap hot; colorbar;
title(sprintf('|u_{num} - u_{exact}|, CFL=%s, %dx%d', CFL_tag, nx, ny));
xlabel('x'); ylabel('y');
if save_figs, save_pair('2d_error_heatmap'); end

%% 
jmid = round((ny+1)/2);
figure('Color','w','Name','Center-line profile');
plot(xv, Ue(jmid,:), '-','LineWidth',1.6); hold on; grid on
plot(xv, Un(jmid,:),'--','LineWidth',1.4);
xlabel('x'); ylabel(sprintf('u(x, y=%.3f, T)', yv(jmid)));
legend('Exact','Numerical','Location','best');
title(sprintf('Profile at y=mid (CFL=%s, %dx%d)', CFL_tag, nx, ny));
if save_figs, save_pair('2d_centerline_profile'); end

%% ================= 4) convergence curve：L2 vs h（loglog） =================
h = G.dx;           
L2 = G.l2_error;

figure('Color','w','Name','Convergence L2 vs h');
loglog(h, L2, 'o-','LineWidth',1.6); hold on; grid on
% 斜率=2参考线
x0=h(1); y0=L2(1); xx=logspace(log10(min(h)),log10(max(h)),200); yy=y0*(xx/x0).^2;
loglog(xx,yy,'--','LineWidth',1.2);
set(gca,'XDir','reverse');
xlabel('h'); ylabel('L_2 error');
legend('GPU (optimized)','slope = 2 ref','Location','southwest');
title(sprintf('Convergence (CFL=%s)', CFL_tag));
if save_figs, save_pair('2d_convergence_L2_vs_h'); end

%% ================= 5) observed order p =================
figure('Color','w','Name','Observed order p');
plot(G.Nx, G.order_p, 'o-','LineWidth',1.6); grid on
xlabel('N_x'); ylabel('observed order p'); yline(2,'--');
title(sprintf('Observed order p (CFL=%s)', CFL_tag));
if save_figs, save_pair('2d_observed_order_p'); end

%% ================= 6) total time vs N_x（loglog） =================
figure('Color','w','Name','Perf total time vs Nx');
loglog(NxG, t_gpu, 's-','LineWidth',1.6); hold on; grid on
xlabel('N_x (grid per side)'); ylabel('Time (s)');
ttl = sprintf('2D Total runtime vs N_x  (CFL=%s, T=%s)', CFL_tag, T_label);
if hasCPU
    loglog(Nx_s, t_cpu, 'o-','LineWidth',1.6);
    legend('GPU total (kernel)','CPU total','Location','northwest');
else
    legend('GPU total (kernel)','Location','northwest');
end
title(ttl);
if save_figs, save_pair('2d_perf_time_vs_Nx'); end

%% ================= 7) Speedup vs N_x（若有 CPU） =================
if hasCPU
    figure('Color','w','Name','Speedup vs Nx');
    plot(Nx_s, speedup, 'o-','LineWidth',1.6); grid on
    xlabel('N_x (grid per side)'); ylabel('Speedup (CPU/GPU)');
    title(sprintf('2D Speedup vs N_x  (CFL=%s, T=%s)', CFL_tag, T_label));
    yline(1,'--','No speedup','LabelHorizontalAlignment','left');
    if save_figs, save_pair('2d_perf_speedup_vs_Nx'); end
end

%% ================= 8) GPU decomposition =================
timing_file = 'timings2D.csv'; 
if isfile(timing_file)
    Tm = readtable(timing_file);
    Tm.Properties.VariableNames = matlab.lang.makeValidName(Tm.Properties.VariableNames);
    need = ["Nx","Ny","h2d_ms","kernel_ms","d2h_ms","total_ms"];
    if all(ismember(need, string(Tm.Properties.VariableNames)))
        % 对齐 (Nx,Ny)：选最大 Nx 的那条
        [Ng_tm, ia, ib] = intersect([Tm.Nx(:) Tm.Ny(:)], [NxG NyG], 'rows');
        if ~isempty(Ng_tm)
            [~, idxMax] = max(Ng_tm(:,1)); % 最大 Nx
            k = ia(idxMax);
            figure('Color','w','Name','GPU breakdown at max Nx');
            bar([Tm.h2d_ms(k), Tm.kernel_ms(k), Tm.d2h_ms(k)]/1000);
            set(gca,'XTickLabel',{'H2D','Kernel','D2H'}); grid on
            ylabel('Time (s)');
            title(sprintf('2D GPU breakdown at Nx=%d, Ny=%d', Tm.Nx(k), Tm.Ny(k)));
            if save_figs, save_pair('2d_perf_gpu_breakdown_maxN'); end
        end
    end
end

%% 
disp('==== Summary (GPU optimized) ====');
S = table(G.Nx, G.Ny, G.dx, G.dy, L2, t_gpu, 'VariableNames',{'Nx','Ny','dx','dy','L2','GPU_s'});
disp(S);
if hasCPU
    disp('==== CPU vs GPU (aligned) ====');
    T = table(Nx_s, Ny_s, t_cpu, t_gpu_s, speedup, ...
        'VariableNames',{'Nx','Ny','CPU_s','GPU_s','Speedup'});
    disp(T);
end

fprintf('Done. 输出图保存在当前目录（若 save_figs=true）。\n');

%% ======= helpers =======
    function name = find_col(T, candidates)
        name = '';
        for kk = 1:numel(candidates)
            if any(strcmpi(T.Properties.VariableNames, candidates(kk)))
                name = candidates(kk); return;
            end
        end
    end

    function save_pair(basename)
        saveas(gcf, [basename '.png']);
        exportgraphics(gcf, [basename '.pdf'],'ContentType','vector');
    end
end