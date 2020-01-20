#!/usr/bin/python
# encoding: utf-8

import numpy as np
import cv2
import glob
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC

def import_image(path):
    """
    INPUT: path to image file in jpg
    OUTPUT: machine readable image file
    """
    image = cv.imread(path)
    return image

def label_imgs(positive_images_path, negative_images_path, image_suffix="*.jpg"):
    """
    INPUT: path to positive images, path to negative images, image suffix
    OUTPUT: numpy array of labeled image paths
    """
    # I use positive and negative here because that's what it is in terms of a model
    # positive means that which is labeled with the label we are trying to predict for
    # and negative is what we are trying to predict against
    # I'm using all the images I have, since I only have a few
    pos_images = set(glob.glob(positive_images_path + '/' + image_suffix))
    neg_images = set(glob.glob(negative_images_path + '/' + image_suffix))

    # truth value - computation is inherently binary from the settings of bits
    # to binaries in model construction 
    labeled_img_paths = [[path, True] for path in positive_imgs] + [[path, False] for path in negative_imgs]
    labeled_image_array = np.array(labeled_image_paths)
    return labeled_image_array

def generate_features(labeled_image_array):
    """
    INPUT: numpy array of labeled image paths
    OUTPUT: SIFT descriptors for the images and the labels
    """
    image_descriptors = []

    for image_path, label in labeled_image_array: 
        image = read_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(gray, None)
        image_descriptors.append(descriptors)

    y = np.array(labeled_image_paths)[:, -1]
    return image_descriptors, y

def generate_cluster_features(image_descriptors, clustering_model):
    """
    INPUT: SIFT descriptors for the images and a clustering model
    OUTPUT: training set
    """
    number_of_clusers = clustering_model.n_clusters
    descriptors_array = np.array(image_descriptors)
    clustering_model.fit(descriptors_array)
    clustered_words = [clustering_model.predict(words) for words in image_descriptors]
    X = np.array([np.bincount(words, min_length=number_of_clusters) for words in clustered_words])
    return X

def main():
    # I built this with the Labels "safe" and "unsafe" in the directory structure
    positive_image_paths = "../images/train/safe"
    negative_image_paths = "../images/train/unsafe"
    labeled_image_array = label_imgs(positive_image_paths, negative_image_paths)
    image_descriptors = generate_features(labeled_image_array)
    clustering = KMeans(n_clusters=5)
    X = generate_cluster_features(image_descriptors, clustering)
    clf = SVC(gamma="auto")
    clf.fit(X, y)
    pickle.dump(clf, open(fs_model.pkl, 'wb')


if __name__ == "__main__":
    main()


