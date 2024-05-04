from enum import auto, Flag

import cv2
import matplotlib.pyplot as plt
import numpy as np


class NoiseType(Flag):
    Gaussian = auto()
    Uniform = auto()


test_img_path = "./test_image.png"
verbose = False


def load_grayscale_img(path: str):

    global verbose

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if verbose:
        print(f"Loaded from path: {path}")
        print(f"Image dimension: {img.shape}")
    return img


def add_noise(img, type: NoiseType):

    gn_img, noise = None, None

    if type == NoiseType.Uniform:
        noise = np.zeros(img.shape, dtype=np.uint8)
        cv2.randu(noise, 0, 255)
        noise = (noise * 0.5).astype(np.uint8)
    elif type == NoiseType.Gaussian:
        noise = np.zeros(img.shape, dtype=np.uint8)
        cv2.randn(noise, 128, 200)
        noise = (noise * 0.5).astype(np.uint8)

    gn_img = cv2.add(img, noise)
    return gn_img, noise


def plot(img, gauss_noise, gn_img):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(gauss_noise, cmap="gray")
    plt.axis("off")
    plt.title("Gaussian Noise")

    fig.add_subplot(1, 3, 3)
    plt.imshow(gn_img, cmap="gray")
    plt.axis("off")
    plt.title("Combined")
    plt.show()


# img = cv2.imread("/kaggle/input/test-image-for-noise/image.jpg", 0)

if __name__ == "__main__":
    verbose = True
    img = load_grayscale_img(test_img_path)
    gn_img, gauss_noise = add_noise(img, type=NoiseType.Gaussian)
    plot(img, gn_img, gauss_noise)
