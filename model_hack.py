#!/usr/bin/python
# encoding: utf-8

import numpy as np
import cv2
import glob
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def import_image(path):
    """
    INPUT: path to image file in jpg
    OUTPUT: machine readable image file
    """
    image = cv2.imread(path)
    return image

class ClusteredImages:
    def __init__(self, positive_images_path, negative_images_path, image_suffix):
        self.positive_images = set(glob.glob(positive_images_path + '/' + image_suffix))
        self.negative_images = set(glob.glob(negative_images_path + '/' + image_suffix))
        self.no_of_clusters = number_of_clusters
    
        self.image_paths = [[path, True] for path in self.positive_images] + [[path, False] for path in self.negative_images]
        self.image_array = np.array(self.image_paths)

        self.descriptors = []

        for path, label in self.image_array:
            image = import_image(path)
            b_and_w = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, each_descriptors = sift.detectAndCompute(b_and_w, None)
            self.descriptors.append(each_descriptors)

    def return_labels(self):
        return np.array(self.image_paths)[:, -1]

    def generate_features(self, clustering_model):
        number_of_clusters = clustering_model.n_clusters
        descriptors_pre_array = [desc for desc_list in self.descriptors for desc in desc_list]
        descriptors_array = np.array(descriptors_pre_array)
        print(descriptors_array)
        
        clustering_model.fit(descriptors_array)
        clustered_words = [clustering_model.predict(words) for words in self.descriptors]
        return np.array([np.bincount(words, minlength=number_of_clusters) for words in clustered_words])

class ParameterFinder:
    def __init__(self, X, y):
        # use gammas for rbf, poly and sigmoid
        #degrees for poly
        # coef0 for poly, sigmoid
        self.X = X
        self.y = y
        self.kernels_to_try = ['linear', 'rbf', 'poly', 'sigmoid']
        self.C_params = [0.001, 0.01, 0.1, 1, 10]
        self.gamma_params = [0.001, 0.01, 0.1, 1]
        self.degree_params = [0.0, 1.0, 2.0, 3.0, 4.0]
       
        def find_best_params(kernel, X, y, param_grid):
            grid_search = GridSearchCV(svm.SVC(kernel = kernel), param_grid)
            grid_search.fit(X, y)
            return grid_search.best_params_

        def return_all_best_params(self):
            best_params = {}
            for kernel in self.kernels_to_try:
                if kernel == 'linear':
                    param_grid = {'C': self.C_params}
                    search_for_params = find_best_params('rbf', self.X, self.y, param_grid)
                    best_params['linear'] = search_for_params
                elif kernel == 'rbf':
                    param_grid = {'C': self.C_params, 'gamma': self.gamma_params}
                    search_for_params = find_best_params('rbf', self.X, self.y, param_grid)
                    best_params['rbf'] = search_for_params
                elif kernel == 'poly':
                    param_grid = {'C': self.C_params, 'gamma': self.gamma_params, 'degree': self.degree_params}
                    search_for_params = find_best_params('poly', self.X, self.y, param_grid)
                    best_params['poly'] = search_for_params
                else: 
                    pass
            return best_params



