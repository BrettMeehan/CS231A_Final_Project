import cv2
import numpy as np
import numpy.matlib
import scipy.misc
import scipy.ndimage
import sys
import os
import detector
sys.path.insert(0, 'filterpy')
from filterpy.kalman import KalmanFilter
from sklearn.svm import LinearSVC

def cross_correlate(image, template):
    h, w = template.shape[:2]

    result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    threshold = 0.85#0.8
    rects = np.empty((1, 4))
    
    #cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    rects[0, :] = np.array([[min_loc[0], min_loc[1], w, h]])

    weights = np.ones(rects.shape[0])
    return (rects, weights)

# use for SIFT or SURF
def SIFT_SURF_bounding_box(detection_obj, image, template):
    kp1, desc1 = detection_obj.detectAndCompute(template, None)
    kp2, desc2 = detection_obj.detectAndCompute(image, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    keypoints1 = np.empty((len(kp1), 2))
    for i in range(len(kp1)):
        keypoints1[i, 0] = kp1[i].pt[0]
        keypoints1[i, 1] = kp1[i].pt[1]
    keypoints2 = np.empty((len(kp2), 2))
    for i in range(len(kp2)):
        keypoints2[i, 0] = kp2[i].pt[0]
        keypoints2[i, 1] = kp2[i].pt[1]

    match_array = np.empty((len(good_matches), 2), dtype=int)
    for i in range(len(good_matches)):
        match_array[i, 0] = good_matches[i][0].queryIdx
        match_array[i, 1] = good_matches[i][0].trainIdx

    (inliers, model) = refine_match(keypoints1, keypoints2, match_array)

    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = template.shape[:2]
    x = int(model[0, 2])
    y = int(model[1, 2])
    scale = int(min(model[0, 0], model[1, 1]))

##############
    '''
    cv2.rectangle(color_img, (x, y), (x + w*scale, y + h*scale), (0, 255, 0), 2)
    cv2.imshow('HI', color_img)
    cv2.waitKey()
    '''
#output = cv2.drawMatchesKnn(template, kp1, image, kp2, good_matches, None,flags=2)
    #cv2.imshow('good matches', output)
    #cv2.waitKey(1000)
##############

    return (np.array([[x, y, w*scale, h*scale]]), np.ones(1))

def refine_match(keypoints1, keypoints2, matches, reprojection_threshold = 10,
                 num_iterations = 1000):
    save_H = None
    save_inliers = None
    save_inlier_count = 0
    
    for i in range(num_iterations):
        point_ind = np.random.choice(matches.shape[0], 3, replace=False)
        A = np.empty([6, 6])
        b = np.empty([6, 1])
        for i in range(3):
            point1 = keypoints1[matches[point_ind[i], 0], :2]
            point2 = keypoints2[matches[point_ind[i], 1], :2]
            A[2*i:2*i+2, :] = np.array([[point1[0], point1[1], 0, 0, 1, 0],
                                        [0, 0, point1[0], point1[1], 0, 1]])
            b[2*i:2*i+2, :] = np.array([[point2[0]], [point2[1]]])
        
        X = np.linalg.lstsq(A, b)[0]
        H = np.array([[X[0, 0], X[1, 0], X[4, 0]],
                      [X[2, 0], X[3, 0], X[5, 0]], [0, 0, 1]])
        matches1_coord = keypoints1[matches[:, 0], :2].T
        matches1_coord = np.concatenate((matches1_coord, np.ones([1, matches1_coord.shape[1]])), axis=0)
        matches2_coord = keypoints2[matches[:, 1], :2].T
        matches2_coord = np.concatenate((matches2_coord,
                                           np.ones([1, matches2_coord.shape[1]])), axis=0)
        transformed_coord = H.dot(matches1_coord)
                  
        diff = transformed_coord - matches2_coord
        dist = np.linalg.norm(diff, axis=0)
                          
        sorted_dist_ind = np.argsort(dist)
        inlier_count = 0
        for j in range(sorted_dist_ind.shape[0]):
                  if dist[sorted_dist_ind[j]] > reprojection_threshold:
                      break
                  inlier_count += 1
                              
        if inlier_count > save_inlier_count:
            save_H = H
            save_inliers = sorted_dist_ind[:inlier_count]
            save_inlier_count = inlier_count

    return (save_inliers, save_H)

def distances_from_predict(x, y, rects):
    point = np.matlib.repmat(np.array([x[0], y[0]]), rects.shape[0], 1)
    dist = np.linalg.norm(point - rects[:, :2], axis=1)
    return dist.flatten()

def train_SVM():
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

    return clf

def update_template(frame, rects):
    # only use first detection as template
    x = rects[0, 0]
    y = rects[0, 1]
    w = rects[0, 2]
    h = rects[0, 3]
    return frame[y:y+h, x:x+w]




def main():
    #####################DETECTION SETTINGS######################
    videos = ['./IMG_1300.m4v', './ped_crossing.mp4', './ped_tire.mp4', './motorcycle_pedestrian.mp4',
              './karma_ped.mp4']
    video_selection = 0
    video_name = videos[video_selection]
    
    use_nms = True # non-maximum suppression
    use_symmetry_score = False # use symmetry score to try to throw away false pedestrian detections
    detection_frequency = 10 # detect every n frames
    use_kalman_filter = True # smooths out detection path and able to predict locations
    
    prune_slow_detections = True # prune detections that don't achieve a certain speed threshold
    min_speed_thresh = 20 # "avg" num. pixels moved per frame in (slow_detection_frame_limit) frames
    slow_detection_frame_limit = 80


    methods = ['HOG', 'SIFT', 'SURF', 'cross-correlation']
    method_selection = 0

    use_opencv_HOG_detector = True # toggles between opencv HOG and the HOG detector I created
    window_size = np.array([72, 36])
    
    trim_SIFT_SURF_template = True
    trim = 10
    
    save_video = False
    #############################################################










    if use_symmetry_score:
        # get typical symmetry score from training data
        symmetry_sigma = 1

        image_files = [os.path.join('data/pedestrian_scenes', f) for f in os.listdir('data/pedestrian_scenes') if (f.endswith('.jpg') or f.endswith('.png'))]
        num_images = len(image_files)

        scores = np.empty(num_images)
        for i in range(num_images):
            img = scipy.misc.imresize(scipy.ndimage.imread(image_files[i], 'L'), window_size)
            score = detector.compute_symmetry_score(img)
            scores[i] = score

        mean_symmetry_score = np.mean(scores)
        std_symmetry_score = np.std(scores)

    if use_kalman_filter:
        k_filters = []
        starting_pos = []
        max_dist = []

    if method_selection > 0:
        template = None
    if method_selection == 1:
        sift = cv2.xfeatures2d.SIFT_create()
    elif method_selection == 2:
        surf = cv2.xfeatures2d.SURF_create()


        '''
        #template = cv2.imread('cropped_silhouette.png')
        template = cv2.imread('found_pedestrian.jpg')
        #cv2.imshow('inital template', template)
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        #cv2.imshow('template', template)
        #cv2.waitKey()
        im = cv2.imread('screen1.png')
        #cv2.imshow('image', im)
        #cv2.waitKey()
        ##im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #cross_correlate(im, template)
        sys.exit(1)
        '''
    ####
    '''
        template = cv2.imread('found_pedestrian.jpg')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image = cv2.imread('screen6.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        find_bounding_box(sift, image, template)
        sys.exit(1)
    '''
    ####
    #cv2.imwrite('sift_keypoints.jpg', img)
    #sys.exit(1)



    if use_opencv_HOG_detector:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    else:
        clf = train_SVM()


    cap = cv2.VideoCapture(video_name)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_name)
        cv2.waitKey(1000)
        print('Wait for the header')

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (w, h))


    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    count = 0
    get_initial_detection = True
    while True:
        flag, frame = cap.read()
        if flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(str(pos_frame) + ' frames')
            if get_initial_detection:# get initial detection
                if use_opencv_HOG_detector:
                    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05)
                    rects = np.asarray(rects)
                else:
                    (rects, weights) = detector.run_detector(frame, clf, window_size)

                if rects.size == 0:
                    continue
                else:
                    get_initial_detection = False
                if use_nms:
                    (rects, weights) = detector.non_max_suppression(rects, weights)
                if use_symmetry_score:
                    rects = detector.remove_false_detections(frame, rects, mean_symmetry_score,
                                                         std_symmetry_score, symmetry_sigma)
                    if rects is None:
                        continue
                if use_kalman_filter:
                    for i in range(rects.shape[0]): # initialize Kalman filters
                        x = rects[i, 0]
                        y = rects[i, 1]
                        w = rects[i, 2]
                        h = rects[i, 3]

                        filter_x = KalmanFilter(dim_x=2, dim_z=1)
                        filter_y = KalmanFilter(dim_x=2, dim_z=1)

                        filter_x.x = np.array([[x, 0.]]).T
                        filter_y.x = np.array([[y, 0.]]).T

                        filter_x.F = np.array([[1., 1],
                                               [0, 1]])
                        filter_y.F = np.array([[1., 1],
                                               [0, 1]])
                        filter_x.P *= 1.
                        filter_y.P *= 1.

                        k_filters.append((filter_x, filter_y, w, h))
                        starting_pos.append((x, y, pos_frame))
                        max_dist.append(0)
                if method_selection > 0:
                    template = update_template(frame, rects)
                    ##########################
                    if (method_selection == 1 or method_selection == 2) and trim_SIFT_SURF_template:
                        template_shape = template.shape
                        template = template[trim:(template_shape[0]-trim), trim:(template_shape[1]-trim)]
                    ##########################
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                for i in range(rects.shape[0]): # draw initial detections
                    x = rects[i, 0]
                    y = rects[i, 1]
                    w = rects[i, 2]
                    h = rects[i, 3]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif pos_frame % detection_frequency == 0: # incorporate new measurements
                                                       # based on detection method
                if method_selection == 0:
                    if use_opencv_HOG_detector:
                        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05)
                        rects = np.asarray(rects)
                    else:
                        (rects, weights) = detector.run_detector(frame, clf, window_size)
                elif method_selection == 1:
                    (rects, weights) = SIFT_SURF_bounding_box(sift, frame, template)
                elif method_selection == 2:
                    (rects, weights) = SIFT_SURF_bounding_box(surf, frame, template)
                elif method_selection == 3:
                    (rects, weights) = cross_correlate(frame, template)
                
                
                
                if rects.size == 0:
                    continue
                if use_nms:
                    (rects, weights) = detector.non_max_suppression(rects, weights)
                if use_symmetry_score:
                    rects = detector.remove_false_detections(frame, rects, mean_symmetry_score,
                                                         std_symmetry_score, symmetry_sigma)
                    if rects is None:
                        continue

                if use_kalman_filter:
                    kalman_bboxes = []
                    kalman_filter_ind = 0
                    for (filter_x, filter_y, w, h) in k_filters: # update Kalman filters
                        filter_x.predict()
                        filter_y.predict()

                        num_measurements = rects.shape[0]
                        filter_x.dim_z = num_measurements
                        filter_y.dim_z = num_measurements

                        filter_x.H = np.matlib.repmat(np.array([[1., 0]]), num_measurements, 1)
                        filter_y.H = np.matlib.repmat(np.array([[1., 0]]), num_measurements, 1)

                        distances = distances_from_predict(filter_x.x[0], filter_y.x[0], rects)

                        R_x = R_y = np.diag(distances**3) # weight confidence/variance based on distance
                                                          # from predicted location

                        z_x = np.array(rects[:, 0])
                        z_y = np.array(rects[:, 1])
                        
                        if len(z_x.shape) < 2:
                            z_x = np.array([z_x]).T
                        if len(z_y.shape) < 2:
                            z_y = np.array([z_y]).T

                        filter_x.update(z=z_x, R=R_x)
                        filter_y.update(z=z_y, R=R_y)

                        x_pos = filter_x.x[0]
                        y_pos = filter_y.x[0]
                        kalman_bboxes.append((x_pos, y_pos))
                    
                        if prune_slow_detections:
                            # update max distance from start
                            dist = np.linalg.norm(np.array([x_pos, y_pos]) -\
                                                  np.array([starting_pos[kalman_filter_ind][0],
                                                            starting_pos[kalman_filter_ind][1]]))
                            if dist > max_dist[kalman_filter_ind]:
                                max_dist[kalman_filter_ind] = dist
                            kalman_filter_ind += 1
                    #if method_selection > 0:
                    #    templates = update_templates(frame, rects)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    for i in range(len(kalman_bboxes)):
                        (x, y) = kalman_bboxes[i]
                        (filter_x, filter_y, w, h) = k_filters[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    #if method_selection > 0:
                    #    templates = update_templates(frame, rects)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    for i in range(rects.shape[0]):
                        x = rects[i, 0]
                        y = rects[i, 1]
                        w = rects[i, 2]
                        h = rects[i, 3]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elif use_kalman_filter: # predict unknown positions
                kalman_bboxes = []
                kalman_filter_ind = 0
                for (filter_x, filter_y, w, h) in k_filters: # update Kalman filters
                    filter_x.predict()
                    filter_y.predict()
                    x_pos = filter_x.x[0]
                    y_pos = filter_y.x[0]
                    kalman_bboxes.append((x_pos, y_pos))
                    
                    if prune_slow_detections:
                        # update max distance from start
                        dist = np.linalg.norm(np.array([x_pos, y_pos]) -\
                                              np.array([starting_pos[kalman_filter_ind][0],
                                                        starting_pos[kalman_filter_ind][1]]))
                        if dist > max_dist[kalman_filter_ind]:
                            max_dist[kalman_filter_ind] = dist
                        kalman_filter_ind += 1
                '''
                if method_selection > 0:
                    for i in range(len(kalman_bboxes)):
                        (x, y) = kalman_bboxes[i]
                        (filter_x, filter_y, w, h) = k_filters[i]
                    templates = update_templates(frame, rects)
                '''
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                for i in range(len(kalman_bboxes)):
                    (x, y) = kalman_bboxes[i]
                    (filter_x, filter_y, w, h) = k_filters[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            ###############
            cv2.imshow('video', frame)
            if save_video:
                out.write(frame)
            ###############
            
            if prune_slow_detections and use_kalman_filter and k_filters:
                deletion_list = []
                for i in range(len(k_filters)):
                    if pos_frame - starting_pos[i][2] == slow_detection_frame_limit:
                        avg_speed = max_dist[i]/float(slow_detection_frame_limit)
                        if avg_speed < min_speed_thresh:
                            deletion_list.append(i)
            
                for i in range(len(deletion_list) - 1, -1, -1):
                    print('REMOVING detection')
                    del k_filters[deletion_list[i]]
                    del starting_pos[deletion_list[i]]
                    del max_dist[deletion_list[i]]

                if not k_filters: # get a new detection if we have pruned all current ones
                    get_initial_detection = True


            ###########
            
            #if pos_frame == 2000:
            #cross_correlate(frame, template)
            
            #find_bounding_box(sift, frame, template, kp1, desc1)
            #    break
                
            ############
            #cv2.waitKey(20000)
            #sys.exit(1)
            ############

        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print('frame not ready')
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == \
        cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break


if __name__ == '__main__':
    main()
