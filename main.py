import cv2
import matplotlib.pyplot as plt
import numpy as np

""
# img = cv2.imread("/kaggle/input/test-image-for-noise/image.jpg", 0)

if __name__ == "__main__":
    img = cv2.imread("./test_image.png", 0)
    print(img.shape)
    print("Hello world")
