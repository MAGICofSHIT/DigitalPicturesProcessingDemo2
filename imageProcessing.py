import numpy as np


def FT(image):
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