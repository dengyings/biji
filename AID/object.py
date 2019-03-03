import os
import sys
import warnings
import numpy as np
import cv2 as cv
import hmmlearn.hmm as hl


def show_image(title, image):
    cv.imshow(title, image)


def search_objects(directory):
    if not os.path.isdir(directory):
        raise IOError(
            "The director '" + directory + "' doesn't exist!")
    objects = {}
    for curdir, subdirs, files in os.walk(directory):
        for jpeg in (file for file in files if file.endswith('.jpg')):
            path = os.path.join(curdir, jpeg)
            label = path.split(os.path.sep)[-2]
            if label not in objects:
                objects[label] = []
            objects[label].append(path)
    return objects


def read_image(filename):
    image = cv.imread(filename)
    return image


def resize_image(image, size):
    h, w = image.shape[:2]
    scale = size / min(h, w)
    image = cv.resize(image, None, fx=scale, fy=scale)
    return image


def calc_features(image):
    star = cv.xfeatures2d.StarDetector_create()
    keypoints = star.detect(image)
    sift = cv.xfeatures2d.SIFT_create()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, desc = sift.compute(gray, keypoints)
    return desc


def read_data(directory):
    objects = search_objects(directory)
    x, y, z = [], [], []
    for label, filenames in objects.items():
        z.append([])
        descs = np.array([])
        for filename in filenames:
            print(filename, '->', label)
            image = read_image(filename)
            z[-1].append(image)
            image = resize_image(image, 200)
            desc = calc_features(image)
            descs = desc if len(descs) == 0 else np.append(descs, desc, axis=0)
        x.append(descs)
        y.append(label)
    return x, y, z


def train_models(x, y):
    models = {}
    for descs, label in zip(x, y):
        model = hl.GaussianHMM(n_components=4, covariance_type='diag',
                               n_iter=1000)
        models[label] = model.fit(descs)
    return models


def pred_models(models, x):
    y = []
    for descs in x:
        best_score, best_label = None, None
        for label, model in models.items():
            score = model.score(descs)
            if best_score is None:
                best_score = score
            if best_label is None:
                best_label = label
            if best_score < score:
                best_score, best_label = score, label
        y.append(best_label)
    return y


def show_labels(labels, pred_labels, images):
    i = 0
    for label, pred_label, row in zip(labels, pred_labels, images):
        for image in row:
            i += 1
            show_image('{}: {} {} {}'.format(
                i, label, '==' if label == pred_label else '!=',
                pred_label), image)


def main(argc, argv, envp):
    warnings.filterwarnings(
        'ignore', category=DeprecationWarning)
    np.seterr(all='ignore')
    train_x, train_y, train_z = read_data('objects\\training')
    test_x, test_y, test_z = read_data('objects\\testing')
    models = train_models(train_x, train_y)
    pred_test_y = pred_models(models, test_x)
    show_labels(test_y, pred_test_y, test_z)
    cv.waitKey()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
