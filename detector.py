import numpy as np
import os
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from utils import *
import sys
import scipy.misc
from plotting import *

import cv2
'''
RUN_DETECTOR Given an image, runs the SVM detector and outputs bounding
boxes and scores

Arguments:
    im - the image matrix

    clf - the sklearn SVM object. You will probably use the 
        decision_function() method to determine whether the object is 
        a face or not.
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    window_size - an array which contains the height and width of the sliding
    	window

    cell_size - each cell will be of size (cell_size, cell_size) pixels

    block_size - each block will be of size (block_size, block_size) cells

    nbins - number of histogram bins

Returns:
    bboxes - D x 4 bounding boxes that tell [xmin ymin width height] per bounding
    	box

    scores - the SVM scores associated with each bounding box in bboxes

You can compute the HoG features using the compute_hog_features() method
that you implemented in PS3. We have provided an implementation in utils.py,
but feel free to use your own implementation. You will use the HoG features
in a sliding window based detection approach.

Recall that using a sliding window is to take a certain section (called the 
window) of the image and compute a score for it. This window then "slides"
across the image, shifting by either n pixels up or down (where n is called 
the window's stride). 

Using a sliding window approach (with stride of block_size * cell_size / 2),
compute the SVM score for that window. If it's greater than 1 (the SVM decision
boundary), add it to the bounding box list. At the very end, after implementing 
nonmaximal suppression, you will filter the nonmaximal bounding boxes out.
'''
def run_detector(im, clf, window_size, cell_size=6, block_size=2, nbins=9, thresh=1):
    stride = (block_size*cell_size)/2
    window_height = window_size[0]
    window_width = window_size[1]
    im_height = im.shape[0]
    im_width = im.shape[1]

    num_windows_x = 1 + (im_width - window_width)/stride
    num_windows_y = 1 + (im_height - window_height)/stride

    bboxes = None
    scores = []

    for i in range(num_windows_y):
        start_row = i*stride
        end_row = start_row + window_height
        for j in range(num_windows_x):
            print('{}/{}    {}/{}'.format(i, num_windows_y, j, num_windows_x))
            start_column = j*stride
            end_column = start_column + window_width
    
            window = im[start_row:end_row, start_column:end_column]
            #print('{}:{}    {}:{}'.format(start_row, end_row, start_column, end_column))
            features = compute_hog_features(window, cell_size, block_size, 
                                            nbins)
                                            
            #print(features.shape)
            if features.shape[1] < 4 or features.shape[0] < 10:
                break
            
            
            SVMscore = clf.decision_function(features.flatten())
            if(SVMscore > thresh):
                bbox = np.array([[start_column, start_row, window_width,
                                  window_height]])
                if bboxes is None:
                    bboxes = bbox
                else:
                    bboxes = np.concatenate((bboxes, bbox), axis=0)
                scores.append(SVMscore[0])

    return (bboxes, scores)
'''
NON_MAX_SUPPRESSION Given a list of bounding boxes, returns a subset that
uses high confidence detections to suppresses other overlapping
detections. Detections can partially overlap, but the
center of one detection can not be within another detection.

Arguments:
    bboxes - ndarray of size (N,4) where N is the number of detections,
        and each row is [x_min, y_min, width, height]
    
    confidences - ndarray of size (N, 1) of the SVM confidence of each bounding
    	box.

    img_size - [height,width] dimensions of the image.

Returns:
    nms_bboxes -  ndarray of size (N, 4) where N is the number of non-overlapping
        detections, and each row is [x_min, y_min, width, height]. Each bounding box
        should not be overlapping significantly with any other bounding box.

In order to get the list of maximal bounding boxes, first sort bboxes by 
confidences. Then go through each of the bboxes in order, adding them to
the list if they do not significantly overlap with any already in the list. 
A significant overlap is if the center of one bbox is in the other bbox.
'''
def non_max_suppression(bboxes, confidences, old_symmetries=None):
    sorted_indices = list(np.argsort(confidences))
    ind = sorted_indices.pop()
    nms_bboxes = np.array(bboxes[ind, :])

    if len(nms_bboxes.shape) == 1:
        nms_bboxes = np.array([nms_bboxes])
    symmetries = None
    if old_symmetries is not None:
        symmetries = [old_symmetries[ind]]
    while len(sorted_indices) > 0:
        ind = sorted_indices.pop()
        center = (bboxes[ind, 0] + bboxes[ind, 2]/2, 
                  bboxes[ind, 1] + bboxes[ind, 3]/2)
        save_bbox = True
        for i in range(nms_bboxes.shape[0]):
            x_min = nms_bboxes[i, 0]
            x_max = x_min + nms_bboxes[i, 2]
            y_min = nms_bboxes[i, 1]
            y_max = y_min + nms_bboxes[i, 3]
            if x_min <= center[0] <= x_max and y_min <= center[1] <= y_max:
                save_bbox = False
                break
        if save_bbox:
            nms_bboxes = np.concatenate((nms_bboxes, 
                                         np.array([bboxes[ind, :]])), axis=0)
            if old_symmetries is not None:
                symmetries.append(old_symmetries[ind])

    return (nms_bboxes, symmetries)

def compute_symmetry_score(img):
    end_row = img.shape[0]/2
    end_col = img.shape[1]/2

    left_side = img[:end_row, :end_col]
    right_side = np.fliplr(img[end_row:2*end_row, end_col:2*end_col])

    return np.sum(abs(left_side - right_side))

def remove_false_detections(img, bboxes, mean_symmetry_score, std_symmetry_score, symmetry_sigma):
    symmetries = []
    for bbox in bboxes:
        symmetries.append(compute_symmetry_score(img[bbox[0]:(bbox[0] + bbox[2]),
                                                     bbox[1]:(bbox[1] + bbox[3])]))
    
    valid_boxes = None
    for i in range(len(symmetries)):
        if symmetries[i] >= mean_symmetry_score - symmetry_sigma*std_symmetry_score and \
           symmetries[i] <= mean_symmetry_score + symmetry_sigma*std_symmetry_score:
            if valid_boxes is None:
                valid_boxes = np.array([bboxes[i, :]])
            else:
                valid_boxes = np.concatenate((valid_boxes,
                                              np.array([bboxes[i, :]])), axis=0)
    return valid_boxes

if __name__ == '__main__':
    block_size = 2
    cell_size = 6
    nbins = 9
    window_size = np.array([72, 36])
    
    
    # get typical symmetry score from training data
    symmetry_sigma = 1
    
    image_files = [os.path.join('data/pedestrian_scenes', f) for f in os.listdir('data/pedestrian_scenes') if (f.endswith('.jpg') or f.endswith('.png'))]
    num_images = len(image_files)
    
    scores = np.empty(num_images)
    for i in range(num_images):
        img = scipy.misc.imresize(imread(image_files[i], 'L'), window_size)
        score = compute_symmetry_score(img)
        scores[i] = score
    
    mean_symmetry_score = np.mean(scores)
    std_symmetry_score = np.std(scores)



    # compute or load features for training
    if not os.path.exists('data/features_pos.npy'):
        features_pos = get_positive_features('data/pedestrian_scenes',
                                             cell_size, window_size, block_size, nbins)
        np.save('data/features_pos.npy', features_pos)
    else:
        features_pos = np.load('data/features_pos.npy')

    if not os.path.exists('data/features_neg.npy'):
        num_negative_examples = 10000
        features_neg = get_random_negative_features('data/non_pedestrian_scenes', cell_size, window_size, block_size, nbins, num_negative_examples)
        np.save('data/features_neg.npy', features_neg)
    else:
        features_neg = np.load('data/features_neg.npy')

    X = np.vstack((features_pos, features_neg))
    Y = np.hstack((np.ones(len(features_pos)), np.zeros(len(features_neg))))

    # Train the SVM
    clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(X, Y)
    score = clf.score(X, Y)



    '''
    test_img1 = scipy.misc.imresize(imread('data/pedestrian_scenes/pose1.jpg', 'L'), window_size)
    test_img2 = scipy.misc.imresize(imread('car.jpg', 'L'), window_size)
    features = compute_hog_features(test_img1, cell_size, block_size, nbins)
    SVMscore = clf.decision_function(features.flatten())
    print(SVMscore)
    plt.imshow(test_img2, cmap='gray')
    plt.show()
    features = compute_hog_features(test_img2, cell_size, block_size, nbins)
    SVMscore = clf.decision_function(features.flatten())
    print(SVMscore)

    #show_hog(test_img2, features, figsize = (18,6))

    sys.exit(1)
    '''




    image = imread('screen4.png', 'L').astype(np.uint8)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                        padding=(8, 8), scale=1.05)
    #nonmaxsuppression()
    plot_img_with_bbox(image, rects, 'opencv')
    plt.show()

    sys.exit(1)



    # Part A: Sliding window detector
    im = imread('screen3.png', 'L').astype(np.uint8)
    im = scipy.misc.imresize(im, (350, 486))
    
    bboxes, scores = run_detector(im, clf, window_size, cell_size, block_size, nbins)
    plot_img_with_bbox(im, bboxes, 'Without nonmaximal suppresion')
    plt.show()

    # Part B: Nonmaximal suppression
    bboxes, symmetries = non_max_suppression(bboxes, scores, symmetries)
    plot_img_with_bbox(im, bboxes, 'With nonmaximal suppresion')
    plt.show()
    
    # without false detections
    bboxes = remove_false_detections(im, bboxes, symmetries, mean_symmetry_score, std_symmetry_score,
                                    symmetry_sigma)
    plot_img_with_bbox(im, bboxes, 'Without bad symmetry scores')
    plt.show()
