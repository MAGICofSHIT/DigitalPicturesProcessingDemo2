import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体为 SimHei
rcParams['font.family'] = 'SimHei'

# 另外可以设置负号显示正确
rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 读取图像并转换为灰度图像
    path = "./Pictures/fingerprint.tif"
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 如果图像读取失败，给出提示
    if image is None:
        print("无法读取图像，请检查图像路径")
        exit()

    # 获取图像的尺寸
    M, N = image.shape

    # 将图像数据类型转换为 float64，以支持负数
    image = image.astype(np.float64)

    # 创建一个与图像大小相同的复数数组来存储傅里叶变换结果
    F = np.zeros((M, N), dtype=complex)

    # 对图像进行中心化处理：乘以 (-1)^(x+y)
    centered_image = np.zeros_like(image, dtype=float)
    for x in range(M):
        for y in range(N):
            centered_image[x, y] = image[x, y] * ((-1) ** (x + y))

    F = np.fft.fft2(centered_image)

    # 计算傅里叶变换
    # for u in range(M):
    #     for v in range(N):
    #         print(f"u={u},v={v}\n")
    #         sum_val = 0
    #         temp_ex = -2j * np.pi * u / M
    #         temp_ey = -2j * np.pi * v / N
    #         for x in range(M):
    #             for y in range(N):
    #                 exponent = x * temp_ex + y * temp_ey
    #                 sum_val += image[x, y] * np.exp(exponent)
    #         F[u, v] = sum_val

    # 频谱移位，使得低频部分移动到中心
    # Fshift = np.fft.fftshift(F)

    # 计算频谱的幅度并取对数，以便更容易可视化
    magnitude_spectrum = np.abs(F)
    magnitude_spectrum = np.log(magnitude_spectrum + 1)

    # 保存频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('频谱图 (频域)')
    plt.axis('off')
    plt.savefig('./Pictures/centered_frequency_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图形
