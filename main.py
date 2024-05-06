from enum import auto, Flag

import admm_denoiser

import cv2
import matplotlib.pyplot as plt
import numpy as np


class NoiseType(Flag):
    Gaussian = auto()
    Uniform = auto()
    Impulse = auto()


# test_img_path = "./test_image.png"
test_img_path = "./reg.png"

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
    elif type == NoiseType.Impulse:
        noise = np.zeros(img.shape, dtype=np.uint8)
        cv2.randu(noise, 0, 255)
        noise = cv2.threshold(noise, 245, 255, cv2.THRESH_BINARY)[1]
        noise = (noise * 0.5).astype(np.uint8)

    gn_img = cv2.add(img, noise)
    return gn_img, noise


def plot(img, gauss_noise, gn_img, solved):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 4, 1)
    plt.imshow(solved, cmap="gray")
    plt.axis("off")
    plt.title("Solved")

    fig.add_subplot(1, 4, 2)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Original")

    fig.add_subplot(1, 4, 3)
    plt.imshow(gauss_noise, cmap="gray")
    plt.axis("off")
    plt.title("Noisy image")

    fig.add_subplot(1, 4, 4)
    plt.imshow(gn_img, cmap="gray")
    plt.axis("off")
    plt.title("Noise")

    plt.show()


# img = cv2.imread("/kaggle/input/test-image-for-noise/image.jpg", 0)

if __name__ == "__main__":
    verbose = True
    img = load_grayscale_img(test_img_path)
    # noise_img, noise = add_noise(img, type=NoiseType.Gaussian)
    noise_img, noise = add_noise(img, type=NoiseType.Uniform)

    denoiser = admm_denoiser.ADMMDenoiserTV(img=noise_img, verbose=True)
    solved = denoiser.solve()
    print(f"Solved: \n{solved}\n")
    print(f"Original: \n{img}\n")
    print(f"Noisy: \n{noise_img}\n")

    plot(img, noise_img, noise, solved)
