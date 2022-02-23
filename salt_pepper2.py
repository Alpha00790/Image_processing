import numpy as np
import random
import cv2
from matplotlib import pyplot as plt


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


image = cv2.imread('images/Alpha.jpg', 0)
noise_img = sp_noise(image, 0.05)
cv2.imwrite('salt_pepper.jpg', noise_img)
plt.subplot(121), plt.imshow(image, cmap="gray")
plt.subplot(122), plt.imshow(noise_img, cmap="gray")
plt.show()
