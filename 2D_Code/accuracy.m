function plot_2d_results()
CFL   = 0.50;               
NxNy  = [513 513];          
LxLy  = [1 1];              
show_figs = true;
err_csv = sprintf('C_error_results_leapfrog2D_CFL%.2f.csv', CFL);
prof_csv = sprintf('profile2D_CFL%.2f_Nx%05d_Ny%05d.csv', CFL, NxNy(1), NxNy(2));

%% ============== Helpers ============================
S_err = safe_read_csv(err_csv);
S_prof = safe_read_csv(prof_csv);

% 从 profile CSV 重构网格/场
xv = unique(S_prof.x);  nx = numel(xv);
yv = unique(S_prof.y);  ny = numel(yv);
assert(nx == NxNy(1) && ny == NxNy(2), 'Nx,Ny 与文件不匹配。');

Ue = reshape(S_prof.u_ex , [nx, ny])';   % 行= y 索引，列= x 索引
Un = reshape(S_prof.u_num, [nx, ny])';
Err = Un - Ue;

[X, Y] = meshgrid(xv, yv);

%% ============== 1) accuracy ==============
if show_figs
    figure('Name','Numerical vs Analytical','Color','w');
    t = get_final_time_from_profile_name(prof_csv); % 可选：从名字解析时间（如果你将来把 t 编进名）
    
    subplot(1,2,1);
    surf(X, Y, Un); shading interp; colormap turbo; colorbar
    title(sprintf('Numerical u(x,y,T), CFL=%.2f, %dx%d', CFL, nx, ny));
    xlabel('x'); ylabel('y'); zlabel('u'); axis tight; set(gca,'YDir','normal'); view(30,35);

    subplot(1,2,2);
    surf(X, Y, Ue); shading interp; colormap turbo; colorbar
    title('Analytical u_{exact}(x,y,T)');
    xlabel('x'); ylabel('y'); zlabel('u'); axis tight; set(gca,'YDir','normal'); view(30,35);
end

%% ============== 2) error field (heatmap) ==========================
if show_figs
    figure('Name','Error field','Color','w');
    imagesc(xv, yv, abs(Err)); axis image; set(gca,'YDir','normal');
    colormap hot; colorbar
    title(sprintf('|u_{num} - u_{exact}|, CFL=%.2f, %dx%d', CFL, nx, ny));
    xlabel('x'); ylabel('y');
end

%% ============== 3) convergence：loglog(L2 vs h) ==================
% 误差 CSV 表头：CFL,Nx,Ny,dx,dy,dt,l1_error,l2_error,relative_l2,order_p,CPU_time(s)
h = S_err.dx;                  
L2 = S_err.l2_error;

figure('Name','Convergence','Color','w');
loglog(h, L2, 'o-','LineWidth',1.6); hold on; grid on
% 斜率=2参考线（过第一点）
x0 = h(1); y0 = L2(1);
xx = logspace(log10(min(h)), log10(max(h)), 200);
yy = y0 * (xx/x0).^2;
loglog(xx, yy, '--','LineWidth',1.2);
set(gca,'XDir','reverse');    % h 变小向右
xlabel('h'); ylabel('L_2 error');
legend('Leapfrog 2D', 'slope = 2 ref','Location','southwest');
title(sprintf('Convergence (CFL=%.2f)', CFL));

%% ============== 4) observed order_p  =======
figure('Name','Observed order p','Color','w');
plot(S_err.Nx, S_err.order_p, 'o-','LineWidth',1.6); grid on
xlabel('Nx'); ylabel('observed order p');
yline(2,'--'); ylim([0, max(2.5, max(S_err.order_p)+0.2)]);
title(sprintf('Observed order p (CFL=%.2f)', CFL));
legend('p from data','p=2 ref','Location','best');

%% ============== 5) profile comparison ===================
jmid = round((ny+1)/2);
figure('Name','Center-line profile','Color','w');
plot(xv, Ue(jmid,:), '-', 'LineWidth',1.6); hold on; grid on
plot(xv, Un(jmid,:),'--','LineWidth',1.4);
xlabel('x'); ylabel(sprintf('u(x, y=%.3f, T)', yv(jmid)));
legend('Exact','Numerical','Location','best');
title(sprintf('Profile at y=mid (CFL=%.2f, %dx%d)', CFL, nx, ny));

%% ============== 6) stability vs CFL ===========
cfl_list = [0.20 0.50 0.90 1.05];
L2_at_max = nan(size(cfl_list));
for k = 1:numel(cfl_list)
    fname = sprintf('C_error_results_leapfrog2D_CFL%.2f.csv', cfl_list(k));
    if isfile(fname)
        T = readtable(fname);
        L2_at_max(k) = T.l2_error(end);
    end
end
figure('Name','Stability vs CFL','Color','w');
bar(cfl_list, L2_at_max); grid on
xlabel('CFL'); ylabel('Final L_2 error at largest grid');
title('Stability vs CFL (CFL>1 应显著劣化)');

fprintf('Done. Figures generated for CFL=%.2f, grid=%dx%d.\\n', CFL, nx, ny);
end

%% ---------- helpers ----------
function s = safe_read_csv(fname)
    if ~isfile(fname), error('File not found: %s', fname); end
    s = readtable(fname);
end

function t = get_final_time_from_profile_name(~)
    
    t = NaN;
end