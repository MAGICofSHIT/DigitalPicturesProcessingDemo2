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
    # F = DFT(image)

    # 计算频谱的幅度
    original_magnitude_spectrum = np.abs(F)

    # 计算相位图
    phase = np.angle(F)

    # 保存原始频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(original_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/original_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    '''对数化'''
    # 对原始频谱取对数
    log_magnitude_spectrum = np.log(original_magnitude_spectrum + 1)

    # 保存频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(log_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/log_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    '''中心化'''
    centered_image = centerImage(image)

    # 计算傅里叶变换
    F_centered = np.fft.fft2(centered_image)

    # 计算频谱的幅度
    centered_magnitude_spectrum = np.abs(F_centered)

    # 计算频谱的幅度并取对数，以便更容易可视化
    log_centered_magnitude_spectrum = np.log(centered_magnitude_spectrum + 1)

    # 保存频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(log_centered_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/log_centered_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    # 计算相位图
    centered_phase = np.angle(F_centered)

    # 保存相位图
    plt.figure(figsize=(6, 6))
    plt.imshow(centered_phase, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/centered_phase.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    '''平移'''
    # 对图像进行平移操作
    translated_image = translate_image(image, 100, 10)

    # 保存平移图像
    cv2.imwrite('./Pictures/translated_fingerprint.png', translated_image)

    # 对图像进行中心化处理：乘以(-1) ^ (x + y)
    centered_translated_image = centerImage(translated_image)

    # 计算傅里叶变换
    F_centered_translated = np.fft.fft2(centered_translated_image)

    # 计算频谱的幅度并取对数，以便更容易可视化
    centered_translated_magnitude_spectrum = np.abs(F_centered_translated)
    log_centered_translated_magnitude_spectrum = np.log(centered_translated_magnitude_spectrum + 1)

    # 保存频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(log_centered_translated_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/translated_log_centered_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    # 计算相位
    translated_phase = np.angle(F_centered_translated)

    # 保存相位图
    plt.figure(figsize=(6, 6))
    plt.imshow(translated_phase, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/translated_centered_phase.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    '''旋转'''
    # 对图像进行旋转操作
    rotated_image = rotate_image(image, 45)

    # 保存平移图像
    cv2.imwrite('./Pictures/rotated_fingerprint.png', rotated_image)

    # 对图像进行中心化处理：乘以(-1) ^ (x + y)
    centered_rotated_image = centerImage(rotated_image)

    # 计算傅里叶变换
    F_centered_rotated = np.fft.fft2(centered_rotated_image)

    # 计算频谱的幅度并取对数，以便更容易可视化
    centered_rotated_magnitude_spectrum = np.abs(F_centered_rotated)
    log_centered_rotated_magnitude_spectrum = np.log(centered_rotated_magnitude_spectrum + 1)

    # 保存频谱图
    plt.figure(figsize=(6, 6))
    plt.imshow(log_centered_rotated_magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/rotated_log_centered_frequency_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    # 计算相位
    rotated_phase = np.angle(F_centered_rotated)

    # 保存相位图
    plt.figure(figsize=(6, 6))
    plt.imshow(rotated_phase, cmap='gray')
    plt.axis('off')
    plt.savefig('./Pictures/rotated_centered_phase.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图形

    '''计算相角对图像信息的贡献（即相角的IDFT）'''
    phase_reconstructed_image = IDFT(M, N, 1, centered_phase)

    # 保存重建图片
    plt.imsave('./Pictures/phase_reconstructed_image.png', phase_reconstructed_image, cmap='gray')

    '''计算幅度谱对图像信息的贡献（即幅度的IDFT）'''
    magnitude_reconstructed_image = IDFT(M, N, centerImage(centered_magnitude_spectrum), 1)

    # 保存重建图片
    plt.imsave('./Pictures/magnitude_reconstructed_image.png', magnitude_reconstructed_image, cmap='gray')