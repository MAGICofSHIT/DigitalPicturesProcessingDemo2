import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from imageProcessing import *

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

    # 计算傅里叶变换
    F = np.fft.fft2(image)
    # F = FT(image)

    # 计算频谱的幅度
    original_magnitude_spectrum = np.abs(F)

    # 频谱移位，使得低频部分移动到中心
    # Fshift = np.fft.fftshift(F)

    # 保存原始频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(original_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/original_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    # 对图像进行中心化处理：乘以(-1) ^ (x + y)
    centered_image = centerImage(image)

    # 计算傅里叶变换
    F_centered = np.fft.fft2(centered_image)

    # 计算频谱的幅度
    centered_magnitude_spectrum = np.abs(F_centered)

    # 保存频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(centered_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/centered_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形


    # 计算频谱的幅度并取对数，以便更容易可视化
    log_centered_magnitude_spectrum = np.log(centered_magnitude_spectrum + 1)

    # 保存频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(log_centered_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/log_centered_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形