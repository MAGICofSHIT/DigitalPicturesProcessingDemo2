clc,clear,close all
% 读取灰度图像
image = imread('./Pictures/fingerprint.tif');
if size(image, 3) == 3
    image = rgb2gray(image); % 转换为灰度图像
end
image = double(image); % 转换为双精度以便计算

% 获取图像尺寸
[M, N] = size(image);

% 初始化傅里叶变换结果矩阵
F = zeros(M, N);

% 实现傅里叶变换
for u = 1:M
    for v = 1:N
        sum_val = 0;
        temp_ex = -2j * pi * (u - 1) / M;
        temp_ey = -2j * pi * (v - 1) / N;
        disp(['u=', num2str(u), ' v=', num2str(v)]);
        for x = 1:M
            for y = 1:N
                exponent = (x - 1) * temp_ex + (y - 1) * temp_ey;
                sum_val = sum_val + image(x, y) * exp(exponent);
            end
        end
        F(u, v) = sum_val;
    end
end

% 频谱移位，将低频成分移到中心
Fshift = fftshift(F);

% 计算幅度谱，并对数缩放
magnitude_spectrum = log(abs(Fshift) + 1);

% 保存频谱图
imwrite(mat2gray(magnitude_spectrum), 'frequency_spectrum_matlab.png');

disp('图像处理完成，已保存原始图像和频谱图');
