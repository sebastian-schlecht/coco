import cv2

import numpy as np
import matplotlib.pyplot as plt


def main():
    mask = np.load("./data/saliency.npy")
    img = cv2.imread("./data/burger.jpg")[16:240, 16:240]

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    plt.imshow(mask)
    plt.show()

    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    main()
