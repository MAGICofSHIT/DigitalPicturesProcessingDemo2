import cv2
import numpy as np
from PIL import Image


def DFT(image):
    """
    计算图像的傅里叶变换。
    参数:
        image (numpy.ndarray): 中心化处理后的灰度图像。
    返回:
        F (numpy.ndarray): 傅里叶变换结果（复数）。
    """
    M, N = image.shape
    F = np.zeros((M, N), dtype=complex)
    for u in range(M):
        for v in range(N):
            sum_val = 0
            for x in range(M):
                for y in range(N):
                    exponent = -2j * np.pi * (u * x / M + v * y / N)
                    sum_val += image[x, y] * np.exp(exponent)
            F[u, v] = sum_val
    return F


def IDFT(M, N, magnitude_spectrum, phase_spectrum):
    """
    使用相角重建图像（假设单位幅值），并手动实现中心化和反傅里叶变换。

    参数:
        phase_spectrum (numpy.ndarray): 输入的相角矩阵。
        save_path (str): 保存重建图像的路径。

    返回:
        reconstructed_image (numpy.ndarray): 反傅里叶变换后的图像。
    """
    # 构造单位幅值的复数频谱
    complex_spectrum = magnitude_spectrum * np.exp(1j * phase_spectrum)  # 复数形式

    # 实现二维反傅里叶变换
    # reconstructed_image = np.zeros_like(magnitude_spectrum, dtype=np.float64)
    # for x in range(M):
    #     for y in range(N):
    #         sum_complex = 0 + 0j  # 初始化复数和
    #         for u in range(M):
    #             for v in range(N):
    #                 angle = 2 * np.pi * ((u * x / M) + (v * y / N))
    #                 sum_complex += complex_spectrum[u, v] * np.exp(1j * angle)  # 直接使用复指数公式
    #         reconstructed_image[x, y] = sum_complex / (N * M)  # 取实部并归一化
    # reconstructed_image = np.abs(reconstructed_image)
    reconstructed_image = np.abs(np.fft.ifft2(complex_spectrum))

    return reconstructed_image


def centerImage(image):
    """
    对图像进行中心化处理，乘以 (-1)^(x+y)
    参数:
        image (numpy.ndarray): 输入的灰度图像。
    返回:
        centered_image (numpy.ndarray): 中心化处理后的图像。
    """
    M, N = image.shape
    centered_image = np.zeros_like(image, dtype=np.float64)
    for x in range(M):
        for y in range(N):
            centered_image[x, y] = image[x, y] * ((-1) ** (x + y))
    return centered_image


def translate_image(image, tx, ty):
    """
    对图像进行平移操作。

    参数:
        image (numpy.ndarray): 输入的灰度图像。
        tx (int): 沿水平方向平移的像素数（正值右移，负值左移）。
        ty (int): 沿垂直方向平移的像素数（正值下移，负值上移）。
    返回:
        translated_image (numpy.ndarray): 平移后的图像。
    """
    rows, cols = image.shape
    # 创建平移矩阵
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    # 应用仿射变换
    translated_image = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
    return translated_image


def rotate_image(image, angle, center=None, scale=1.0):
    """
    对图像进行旋转，并以旋转中心为中心裁切到原始大小，边界填充为白色。

    参数:
        image (numpy.ndarray): 输入的灰度图像。
        angle (float): 旋转角度（以度为单位，正值为逆时针方向）。
        scale (float): 缩放比例，默认为 1.0。
    返回:
        cropped_image (numpy.ndarray): 旋转并裁切到原始大小的图像。
    """
    rows, cols = image.shape

    # 计算旋转角度的弧度值
    angle_rad = np.radians(angle)

    # 计算旋转后的图像宽高
    new_width = int(abs(rows * np.sin(angle_rad)) + abs(cols * np.cos(angle_rad)))
    new_height = int(abs(rows * np.cos(angle_rad)) + abs(cols * np.sin(angle_rad)))

    # 计算新的中心点
    new_center = (new_width // 2, new_height // 2)

    # 创建旋转矩阵，并调整中心点到新图像中心
    rotation_matrix = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, scale)
    rotation_matrix[0, 2] += new_center[0] - cols // 2
    rotation_matrix[1, 2] += new_center[1] - rows // 2

    # 应用仿射变换，输出图像大小调整为新宽高，边界填充为白色
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))

    # 计算裁剪区域，以原始中心为基准裁切原始大小
    start_x = new_center[0] - cols // 2
    start_y = new_center[1] - rows // 2
    cropped_image = rotated_image[start_y:start_y + rows, start_x:start_x + cols]

    return cropped_image
