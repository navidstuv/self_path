"""
Normalize a patch stain to the target image using the method of:
E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

import cv2 as cv
import numpy as np
from data import stain_utils as ut


### Some functions ###


def lab_split(I):
    """
    Convert from RGB uint8 to LAB and split into channels
    :param I: uint8
    :return:
    """
    I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    I = I.astype(np.float32)
    I1, I2, I3 = cv.split(I)
    I1 /= 2.55
    I2 -= 128.0
    I3 -= 128.0
    return I1, I2, I3


def merge_back(I1, I2, I3):
    """
    Take seperate LAB channels and merge back to give RGB uint8
    :param I1:
    :param I2:
    :param I3:
    :return:
    """
    I1 *= 2.55
    I2 += 128.0
    I3 += 128.0
    I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    return cv.cvtColor(I, cv.COLOR_LAB2RGB)


def get_mean_std(I):
    """
    Get mean and standard deviation of each channel
    :param I: uint8
    :return:
    """
    I1, I2, I3 = lab_split(I)
    m1, sd1 = cv.meanStdDev(I1)
    m2, sd2 = cv.meanStdDev(I2)
    m3, sd3 = cv.meanStdDev(I3)
    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return means, stds


### Main class ###
def get_stain_matrix(I, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = (OD[(OD > beta).any(axis=1), :])
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return ut.normalize_rows(HE)
class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        target = ut.standardize_brightness(target)
        means, stds = get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        II = ut.standardize_brightness(I)
        I1, I2, I3 = lab_split(II)
        means, stds = get_mean_std(II)
        if stds[0]==0 or stds[1]==0 or stds[2]==0:
            return I
        else:
            norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
            norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
            norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
            return merge_back(norm1, norm2, norm3)
    def hematoxylin(self, I):
        I = ut.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H

