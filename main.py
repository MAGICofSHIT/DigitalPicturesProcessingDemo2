import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体为 SimHei
rcParams['font.family'] = 'SimHei'

# 另外可以设置负号显示正确
rcParams['axes.unicode_minus'] = False

# 读取图像并转换为灰度图像
path = "./Pictures/fingerprint.tif"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if __name__ == "__main__":
    # 如果图像读取失败，给出提示
    if image is None:
        print("无法读取图像，请检查图像路径")
        exit()

    # 对图像进行傅里叶变换
    f = np.fft.fft2(image)

    # 将频谱移到图像中心
    fshift = np.fft.fftshift(f)

    # 计算频谱的幅度并取对数，以便更容易可视化
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = np.log(magnitude_spectrum + 1)

    # 绘制原始图像和频谱图
    plt.figure(figsize=(12, 6))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像 (空域)')
    plt.axis('off')

    # 显示频谱图
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('频谱图 (频域)')
    plt.axis('off')

    plt.savefig('./Pictures/frequency_spectrum.png')
    plt.show()

    # 反向傅里叶变换，恢复图像
    f_ishift = np.fft.ifftshift(fshift)  # 移回原位
    img_back = np.fft.ifft2(f_ishift)  # 反向傅里叶变换
    img_back = np.abs(img_back)  # 取绝对值

    # 显示恢复后的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(img_back, cmap='gray')
    plt.title('反向傅里叶变换恢复的图像')
    plt.axis('off')
    plt.show()
