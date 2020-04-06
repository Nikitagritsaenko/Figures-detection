import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from skimage import io
from scipy.ndimage.morphology import binary_fill_holes
from skimage.util import img_as_ubyte
from skimage.morphology import binary_opening
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
directory = "LabCV/"
f_in_lines = open(os.path.join(directory, "001_line_in.txt"), "r")
image_src_lines = plt.imread(os.path.join(directory, "001_line_src.png"))
image_src_noise = plt.imread(os.path.join(directory, "001_noise_src.png"))
image_src_pure = plt.imread(os.path.join(directory, "001_pure_src.png"))
image_gt = plt.imread(os.path.join(directory, "001_line_gt.png"))
image_300_200 = plt.imread(os.path.join(directory, "test300_200.png"))

def find_contours(im):
    res = im.copy()
    for x in range(1, im.shape[0]):
        for y in range(im.shape[1]):
            res[x, y] = 0

    q = []
    for x in range(1, im.shape[0]):
        for y in range(im.shape[1]):
            if im[x, y] != 0:
                q.append((x, y))
                while len(q) != 0:
                    elem = q.pop()
                    res[elem[0], elem[1]] = 1
                    for u in range(elem[0] - 1, elem[0] + 2):
                        for v in range(elem[1] - 1, elem[1] + 2):
                            if im[u, v] != 0:
                                im[u, v] = 0
                                if not (((u, v) in q) or ((u, v) in res)):
                                    q.append((u, v))
                return res

    return res


test_img = image_src_lines.copy()
# test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
#res = find_contours(test_img)
threshold = threshold_otsu(test_img)
test_img = test_img > threshold

test_img = binary_fill_holes(test_img)
kernel = np.ones((5, 5), np.uint8)
test_img = binary_opening(test_img)

plt.figure(1, figsize=(15, 15))
imshow(test_img, cmap='gray')
plt.show()