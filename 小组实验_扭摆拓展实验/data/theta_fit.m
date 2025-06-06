% 读取数据
data = readmatrix('data.csv');
t = data(:,1);
y_obs = data(:,2);

% 初始参数估计
y0_0 = mean(y_obs);
A_0 = (max(y_obs)-min(y_obs))/2;

% 通过FFT估计频率
Fs = 1/(t(2)-t(1));
N = length(t);
Y = fft(y_obs - y0_0);
P2 = abs(Y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);
frequencies = Fs*(0:N/2)/N;
[~, idx] = max(P1(2:end));
w_0 = 2*pi*frequencies(idx+1);
p_0 = 0;

initialParams = [y0_0, A_0, w_0, p_0];

% 定义模型和约束
model = @(params, t) params(1) + params(2)*sin(params(3)*t + params(4));
lb = [-inf, 0, 0, -inf];
ub = [inf, inf, inf, inf];

% 优化选项
options = optimoptions('lsqcurvefit',...
    'Algorithm', 'levenberg-marquardt',...
    'MaxIterations', 1000,...
    'Display', 'final');

% 执行拟合
fittedParams = lsqcurvefit(model, initialParams, t, y_obs, lb, ub, options);

% 提取参数
y0_fit = fittedParams(1);
A_fit = fittedParams(2);
w_fit = fittedParams(3);
p_fit = fittedParams(4);

% 计算拟合指标
y_fit = model(fittedParams, t);
residuals = y_obs - y_fit;
SSres = sum(residuals.^2);
SStot = sum((y_obs - mean(y_obs)).^2);
R_squared = 1 - SSres/SStot;

% 创建综合结果窗口
figure('Position', [100 100 900 600], 'Name', '综合拟合结果')

% 主结果图
subplot(2,2,[1 3]);
plot(t, y_obs, 'b.', t, y_fit, 'r-', 'LineWidth', 1.5);
title('正弦曲线拟合结果');
xlabel('时间 t');
ylabel('y 值');
grid on;
legend('观测数据', '拟合曲线', 'Location', 'best');

% 残差图
subplot(2,2,2);
plot(t, residuals, 'g');
title('拟合残差');
xlabel('时间 t');
ylabel('残差');
grid on;

% 参数显示区域
subplot(2,2,4);
axis off;
text(0.1, 0.8, sprintf(['拟合参数:\n'...
    'y₀ = %.4f\n'...
    '振幅 A = %.4f\n'...
    '角频率 ω = %.4f rad/unit\n'...
    '相位 φ = %.4f rad\n'...
    '频率 f = %.4f Hz\n'...
    'R² = %.4f'],...
    y0_fit, A_fit, w_fit, p_fit,...
    w_fit/(2*pi), R_squared),...
    'FontSize', 10,...
    'VerticalAlignment', 'top');

% 调整子图间距
set(gcf, 'Color', 'w');
h = findobj(gcf, 'Type', 'axes');
set(h, 'FontSize', 9, 'LineWidth', 1);