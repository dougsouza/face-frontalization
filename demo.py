import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt


this_path = os.path.dirname(os.path.abspath(__file__))


def demo():
    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    check.check_dlib_landmark_weights()
    # load detections performed by dlib library on 3D model and Reference Image
    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
    # load query image
    img = cv2.imread("test.jpg", 1)
    plt.title('Query Image')
    plt.imshow(img[:, :, ::-1])
    # extract landmarks from the query image
    # list containing a 2D array with points (x, y) for each face detected in the query image
    lmarks = feature_detection.get_landmarks(img)
    plt.figure()
    plt.title('Landmarks Detected')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1])
    # perform camera calibration according to the first face detected
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
    # load mask to exclude eyes from symmetry
    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    # perform frontalization
    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    plt.figure()
    plt.title('Frontalized no symmetry')
    plt.imshow(frontal_raw[:, :, ::-1])
    plt.figure()
    plt.title('Frontalized with soft symmetry')
    plt.imshow(frontal_sym[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    demo()