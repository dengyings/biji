import os
import sys
import platform
import numpy as np
import cv2 as cv
import matplotlib.pyplot as mp
import mpl_toolkits.axes_grid1 as mg


def read_image(filename):
    image = cv.imread(filename)
    return image


def show_image(title, image):
    cv.imshow(title, image)


def calc_features(image):
    star = cv.xfeatures2d.StarDetector_create()
    keypoints = star.detect(image)
    sift = cv.xfeatures2d.SIFT_create()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, desc = sift.compute(gray, keypoints)
    return desc


def draw_desc(desc):
    ma = mp.matshow(desc, cmap='jet')
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('DESC', fontsize=20)
    mp.xlabel('Feature', fontsize=14)
    mp.ylabel('Sample', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(8))
    ax.xaxis.set_minor_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator(8))
    ax.yaxis.set_minor_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, right=True,
                   labeltop=False, labelbottom=True, labelsize=10)
    dv = mg.make_axes_locatable(ax)
    ca = dv.append_axes('right', '3%', pad='3%')
    cb = mp.colorbar(ma, cax=ca)
    cb.set_label('DESC', fontsize=14)


def show_chart():
    mp.show()


def main(argc, argv, envp):
    original = read_image('none.jpg')
    show_image('Original', original)
    desc = calc_features(original)
    draw_desc(desc)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
