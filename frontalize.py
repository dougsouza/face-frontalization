__author__ = 'Douglas'

import scipy.io as scio
import cv2
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})


class ThreeD_Model:
    def __init__(self, path, name):
        self.load_model(path, name)

    def load_model(self, path, name):
        model = scio.loadmat(path)[name]
        self.out_A = np.asmatrix(model['outA'][0, 0], dtype='float32') #3x3
        self.size_U = model['sizeU'][0, 0][0] #1x2
        self.model_TD = np.asarray(model['threedee'][0,0], dtype='float32') #68x3
        self.indbad = model['indbad'][0, 0]#0x1
        self.ref_U = np.asarray(model['refU'][0,0])



def frontalize(img, proj_matrix, ref_U, eyemask):
    ACC_CONST = 800
    img = img.astype('float32')
    print "query image shape:", img.shape

    bgind = np.sum(np.abs(ref_U), 2) == 0
    # count the number of times each pixel in the query is accessed
    threedee = np.reshape(ref_U, (-1, 3), order='F').transpose()
    temp_proj = proj_matrix * np.vstack((threedee, np.ones((1, threedee.shape[1]))))
    temp_proj2 = np.divide(temp_proj[0:2, :], np.tile(temp_proj[2, :], (2,1)))

    bad = np.logical_or(temp_proj2.min(axis=0) < 1, temp_proj2[1, :] > img.shape[0])
    bad = np.logical_or(bad, temp_proj2[0, :] > img.shape[1])
    bad = np.logical_or(bad, bgind.reshape((-1), order='F'))
    bad = np.asarray(bad).reshape((-1), order='F')

    nonbadind = np.nonzero(bad == 0)[0]
    temp_proj2 = temp_proj2[:, nonbadind]
    # because python arrays are zero indexed
    temp_proj2 -= 1
    ind = np.ravel_multi_index((np.asarray(temp_proj2[1, :].round(), dtype='int64'), np.asarray(temp_proj2[0, :].round(),
                                dtype='int64')), dims=img.shape[:-1], order='F')
    synth_frontal_acc = np.zeros(ref_U.shape[:-1])
    ind_frontal = np.arange(0, ref_U.shape[0]*ref_U.shape[1])
    ind_frontal = ind_frontal[nonbadind]
    c, ic = np.unique(ind, return_inverse=True)
    bin_edges = np.r_[-np.Inf, 0.5 * (c[:-1] + c[1:]), np.Inf]
    count, bin_edges = np.histogram(ind, bin_edges)
    synth_frontal_acc = synth_frontal_acc.reshape(-1, order='F')
    synth_frontal_acc[ind_frontal] = count[ic]
    synth_frontal_acc = synth_frontal_acc.reshape((320, 320), order='F')
    synth_frontal_acc[bgind] = 0
    synth_frontal_acc = cv2.GaussianBlur(synth_frontal_acc, (15, 15), 30., borderType=cv2.BORDER_REPLICATE)
    frontal_raw = np.zeros((102400, 3))
    frontal_raw[ind_frontal, :] = cv2.remap(img, temp_proj2[0, :].astype('float32'), temp_proj2[1, :].astype('float32'), cv2.INTER_CUBIC)
    frontal_raw = frontal_raw.reshape((320, 320, 3), order='F')

    # which side has more occlusions?
    midcolumn = np.round(ref_U.shape[1]/2)
    sumaccs = synth_frontal_acc.sum(axis=0)
    sum_left = sumaccs[0:midcolumn].sum()
    sum_right = sumaccs[midcolumn+1:].sum()
    sum_diff = sum_left - sum_right

    if np.abs(sum_diff) > ACC_CONST: # one side is ocluded
        ones = np.ones((ref_U.shape[0], midcolumn))
        zeros = np.zeros((ref_U.shape[0], midcolumn))
        if sum_diff > ACC_CONST: # left side of face has more occlusions
            weights = np.hstack((zeros, ones))
        else: # right side of face has more occlusions
            weights = np.hstack((ones, zeros))
        weights = cv2.GaussianBlur(weights, (33, 33), 60.5, borderType=cv2.BORDER_REPLICATE)

        # apply soft symmetry to use whatever parts are visible in ocluded side
        synth_frontal_acc /= synth_frontal_acc.max()
        weight_take_from_org = 1. / np.exp(0.5+synth_frontal_acc)
        weight_take_from_sym = 1 - weight_take_from_org

        weight_take_from_org = np.multiply(weight_take_from_org, np.fliplr(weights))
        weight_take_from_sym = np.multiply(weight_take_from_sym, np.fliplr(weights))

        weight_take_from_org = np.tile(weight_take_from_org.reshape(320, 320, 1), (1, 1, 3))
        weight_take_from_sym = np.tile(weight_take_from_sym.reshape(320, 320, 1), (1, 1, 3))
        weights = np.tile(weights.reshape(320, 320, 1), (1, 1, 3))

        denominator = weights + weight_take_from_org + weight_take_from_sym
        frontal_sym = np.multiply(frontal_raw, weights) + np.multiply(frontal_raw, weight_take_from_org) + np.multiply(np.fliplr(frontal_raw), weight_take_from_sym)
        frontal_sym = np.divide(frontal_sym, denominator)

        # exclude eyes from symmetry
        frontal_sym = np.multiply(frontal_sym, 1-eyemask) + np.multiply(frontal_raw, eyemask)
        frontal_raw[frontal_raw > 255] = 255
        frontal_raw[frontal_raw < 0] = 0
        frontal_raw = frontal_raw.astype('uint8')
        frontal_sym[frontal_sym > 255] = 255
        frontal_sym[frontal_sym < 0] = 0
        frontal_sym = frontal_sym.astype('uint8')
    else: # both sides are occluded pretty much to the same extent -- do not use symmetry
        frontal_sym = frontal_raw
    return frontal_raw, frontal_sym