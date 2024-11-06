import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def median_filter_numpy(image, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("The kernel size must not be even.")
    
    padded_image = np.pad(image, kernel_size // 2, mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.median(neighborhood)
    
    return filtered_image

def median_filter_opencv(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

if __name__ == "__main__":
    image = cv2.imread("lena.png")
    kernel_sizes = [3, 7, 15, 31]
    
    numpy_image = None
    cv_image = None
    numpy_time = []
    cv_time = []

    for size in tqdm(kernel_sizes):
        start = time()
        numpy_image = median_filter_numpy(image, size)
        numpy_time.append(time() - start)

        start = time()
        cv_image = median_filter_opencv(image, size)
        cv_time.append(time() - start)


    fig = plt.figure(figsize=(5.5, 3.5), layout="constrained")
    spec = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(spec[0, 0])
    ax0.imshow(image)
    ax0.set_title("Оригинальное изображение")

    ax1 = fig.add_subplot(spec[0, 1])
    ax1.imshow(numpy_image)
    ax1.set_title(f"Обработанное самописным фильтром (ядро {kernel_sizes[-1]})")

    ax2 = fig.add_subplot(spec[0, 2])
    ax2.imshow(cv_image)
    ax2.set_title(f"Обработанное фильтром из OpenCV (ядро {kernel_sizes[-1]})")

    ax_all = fig.add_subplot(spec[1, :])
    sizes = [f'Ядро: {s}' for s in kernel_sizes]
    x = np.arange(len(sizes))
    width = 0.1

    multiplier = 0
    bar_dict = {
        "Самопись" : numpy_time,
        "OpenCV" : cv_time
    }

    for attribute, measurement in bar_dict.items():
        offset = width * multiplier
        rects = ax_all.bar(x + offset, measurement, width, label=attribute)
        ax_all.bar_label(rects, padding=3)
        multiplier += 1

    ax_all.set_xticks(x + width, sizes)
    ax_all.set_ylabel('Время (s)')
    ax_all.set_title('Время работы алгоритма')
    ax_all.legend(loc='upper left', ncols=4)

    plt.show()