%% main_circle
% 对 A 点和 B 点运动轨迹进行联合非线性最小二乘拟合
% 模型：
%   x(t) = xc + a*cos(wg*t + phi1) + R*cos(theta(t) + phi2);
%   y(t) = yc + b*sin(wg*t + phi3) + R*sin(theta(t) + phi4);
% 要求残差 < 0.1 且 R^2 > 0.99

clear; clc;

%% 1. 读取数据
data = readmatrix('data_circle.csv');
t   = data(:,1);
xA  = data(:,3);
yA  = data(:,4);
xB  = data(:,5);
yB  = data(:,6);

%% 2. 定义已知 theta(t)
theta = -0.9045 + 0.4808*sin(0.7048*t - 0.6891);

%% 3A. 傅里叶分析估计 wgA（点 A）
Fs = 1/mean(diff(t));  % 采样频率估计
N  = length(t);
XA_fft = fft(xA - mean(xA));
P2A = abs(XA_fft/N);
P1A = P2A(1:floor(N/2)+1);
f  = Fs*(0:floor(N/2))/N;
[~, idxA] = max(P1A(2:end));
f0A = f(idxA+1);
wgA_est = 2*pi*f0A;
fprintf('估计 wgA = %.4f rad/s\n', wgA_est);

%% 3B. 傅里叶分析估计 wgB（点 B）
XB_fft = fft(xB - mean(xB));
P2B = abs(XB_fft/N);
P1B = P2B(1:floor(N/2)+1);
[~, idxB] = max(P1B(2:end));
f0B = f(idxB+1);
wgB_est = 2*pi*f0B;
fprintf('估计 wgB = %.4f rad/s\n', wgB_est);

%% 4. 初始参数估计（A 点）
xc0 = mean(xA);
yc0 = mean(yA);
a0  = (max(xA-xc0) - min(xA-xc0))/2;
b0  = (max(yA-yc0) - min(yA-yc0))/2;
phi10 = 0; phi20 = 0; phi30 = 0; phi40 = 0;
res_x = xA - (xc0 + a0*cos(wgA_est*t + phi10));
R0  = (max(res_x) - min(res_x))/2;
p0_A = [xc0, a0, wgA_est, phi10, yc0, b0, phi30, R0, phi20, phi40];

%% 5. 拟合 A 点
modelA = @(p, tt) [ ...
    p(1) + p(2)*cos(p(3)*tt + p(4)) + p(8)*cos(theta + p(9)); ...
    p(5) + p(6)*sin(p(3)*tt + p(7)) + p(8)*sin(theta + p(10)) ...
];

yA_concat = [xA; yA];

options = optimoptions('lsqcurvefit', 'Display', 'iter', 'TolFun', 1e-12);

lb = [-Inf, 0, 0.9*wgA_est, -2*pi, -Inf, 0, -2*pi, -Inf, -2*pi, -2*pi];
ub = [ Inf, Inf, 1.9*wgA_est,  2*pi,  Inf, Inf,  2*pi,  Inf,  2*pi,  2*pi];

[pA, resnormA, residualA] = lsqcurvefit(modelA, p0_A, t, yA_concat, lb, ub, options);

% 计算 R^2 与最大残差
yA_fit = modelA(pA, t);
resA = yA_concat - yA_fit;
R2_A = 1 - var(resA)/var(yA_concat);
maxResA = max(abs(resA));
fprintf('A 点拟合: R^2 = %.5f, max residual = %.5f\n', R2_A, maxResA);

%% 6. 初始参数估计（B 点）
xc0 = mean(xB);
yc0 = mean(yB);
a0  = (max(xB-xc0) - min(xB-xc0))/2;
b0  = (max(yB-yc0) - min(yB-yc0))/2;
phi10 = 0; phi20 = 0; phi30 = 0; phi40 = 0;
res_x = xB - (xc0 + a0*cos(wgB_est*t + phi10));
R0  = (max(res_x) - min(res_x))/2;
p0_B = [xc0, a0, wgB_est, phi10, yc0, b0, phi30, R0, phi20, phi40];

%% 7. 拟合 B 点
modelB = @(p, tt) [ ...
    p(1) + p(2)*cos(p(3)*tt + p(4)) + p(8)*cos(theta + p(9)); ...
    p(5) + p(6)*sin(p(3)*tt + p(7)) + p(8)*sin(theta + p(10)) ...
];

yB_concat = [xB; yB];

lb = [-Inf, 0, 0.9*wgB_est, -2*pi, -Inf, 0, -2*pi, -Inf, -2*pi, -2*pi];
ub = [ Inf, Inf, 1.1*wgB_est,  2*pi,  Inf, Inf,  2*pi,  Inf,  2*pi,  2*pi];

[pB, resnormB, residualB] = lsqcurvefit(modelB, p0_B, t, yB_concat, lb, ub, options);

yB_fit = modelB(pB, t);
resB = yB_concat - yB_fit;
R2_B = 1 - var(resB)/var(yB_concat);
maxResB = max(abs(resB));
fprintf('B 点拟合: R^2 = %.5f, max residual = %.5f\n', R2_B, maxResB);

%% 8. 结果输出
fprintf('\nA 点参数 pA = [xc1, a1, wg1, phi1, yc1, b1, phi3, R1, phi2, phi4]:\n'); disp(pA);
fprintf('B 点参数 pB = [xc2, a2, wg2, phi1, yc2, b2, phi3, R2, phi2, phi4]:\n'); disp(pB);

% 若需查看拟合曲线对比，可取消以下注释
figure; subplot(2,1,1);
plot(t, xA, '.', t, yA_fit(1:length(t)), '-'); legend('xA 数据','xA 拟合');
subplot(2,1,2);
plot(t, yA, '.', t, yA_fit(length(t)+1:end), '-'); legend('yA 数据','yA 拟合');

% 同理可绘制 B 点结果
figure; subplot(2,1,1);
plot(t, xB, '.', t, yB_fit(1:length(t)), '-'); legend('xB 数据','xB 拟合');
subplot(2,1,2);
plot(t, yB, '.', t, yB_fit(length(t)+1:end), '-'); legend('yB 数据','yB 拟合');
