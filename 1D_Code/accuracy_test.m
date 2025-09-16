%% 0) function
function s = safe_read_csv(fname)
    if ~isfile(fname), error('File not found: %s', fname); end
    s = readtable(fname);
end
%% 1) convergence plot（固定 CFL=0.50，L2 vs dx，loglog + 斜率=2参考线）
S = safe_read_csv('error_results_leapfrog_CFL0.50.csv');

x = S.dx;            % dx
y = S.l2_error;      % L2
figure; loglog(x, y, 'o-','LineWidth',1.3); hold on; grid on
% 斜率=2参考线（过第一点）
x0 = x(1); y0 = y(1);
xx = linspace(min(x), max(x), 200);
yy = y0 * (xx/x0).^2;
loglog(xx, yy, '--','LineWidth',1.0);
set(gca,'XDir','reverse'); % dx 变小向右
xlabel('\Deltax'); ylabel('L_2 error');
legend('Leapfrog (CFL=0.50)','ref slope = 2','Location','southwest');
title('Convergence: L_2 error vs \Deltax (CFL=0.50)');

 %% 2) 数值 vs 解析 profile comparison（取最大N的profile）
Profile = safe_read_csv('profile_CFL0.50_N02048.csv'); 
figure; plot(Profile.x, Profile.u_ex, '-','LineWidth',1.4); hold on; grid on
plot(Profile.x, Profile.u_num,'--','LineWidth',1.2);
xlabel('x'); ylabel('u(x,T)');
legend('Exact','Numerical','Location','best');
title('Profile at final time: numerical vs exact (CFL=0.50, N=1024)');

%% 3) CFL stability plot（固定最大N，比较不同CFL的最终L2误差）
cfl_list = [0.20 0.50 0.90 1.05];
L2_at_maxN = nan(size(cfl_list));
for i = 1:numel(cfl_list)
    fname = sprintf('error_results_leapfrog_CFL%.2f.csv', cfl_list(i));
    T = safe_read_csv(fname);
    % 取最后一行（最大N）
    L2_at_maxN(i) = T.l2_error(end);
end
figure; 
bar(cfl_list, L2_at_maxN); grid on
xlabel('CFL'); ylabel('Final L_2 error at largest N');
title('Stability vs CFL (smaller is stable; >1 should blow up)');

%% 4) convergence order p 可视化（来自 CSV 的 order_p 列）
figure; plot(S.N, S.order_p, 'o-','LineWidth',1.3); grid on
xlabel('N'); ylabel('observed order p');
title('Observed order p (CFL=0.50)');
yline(2,'--'); legend('p from data','p=2 ref','Location','best');





