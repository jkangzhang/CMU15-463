from Eulerian import EulerianVideoMagnification
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

class EulerianTest:
    def __init__(self):
        self.eulerian = EulerianVideoMagnification()

    def test_getFilteredImage(self, image):
        return self.eulerian._getFilteredImage(image, 90, 0.83/2, 1.0/2, 30)


if __name__ == '__main__':
    test = EulerianTest()
    imgPath = sys.argv[1]
    image = cv2.imread(imgPath, 0)
    image = np.float32(image)
    # cv2.imwrite("what.png", image)
    ret = test.test_getFilteredImage(image)
    image += ret * 1.9
    cv2.imwrite("bb.png", ret)
    plt.imshow(image, cmap="gray")
    plt.show()
    # cv2.imwrite("ret.png", ret)
