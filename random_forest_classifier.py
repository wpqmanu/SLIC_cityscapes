from sklearn.ensemble import RandomForestClassifier
import sys
import scipy
import scipy.linalg
import random
import math
import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import cPickle
from PIL import Image
from datetime import datetime
import collections
from collections import Counter
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
# sys.path.insert(0,'/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
import cv2

sys.path.append( os.path.normpath( os.path.join('/home/panquwang/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
from labels     import trainId2label,id2label
# matplotlib.use('Qt4Agg')


def test_lower_bound(all_feature_data):
    data=np.asarray(all_feature_data[0])
    label=np.asarray(all_feature_data[1])

    # majority voting
    data_predictions=data[:,[38,40,42]]
    data_predictions_output=[]
    for temp_index in range(data_predictions.shape[0]):
        value=list(data_predictions[temp_index])
        data_predictions_output.append(max(set(value), key=value.count))

    lower_bound = float(np.sum(np.asarray(data_predictions_output) == label)) / len(label)
    print "Accuracy of lower bound is " + str(lower_bound)


def random_forest_classifier(all_feature_data):
    data=np.asarray(all_feature_data[0])
    label=np.asarray(all_feature_data[1])

    # data=sklearn.preprocessing.normalize(data,axis=0)

    clf = RandomForestClassifier(n_estimators=50, max_depth=None,min_samples_split=2,verbose=1)
    fit_clf=clf.fit(data,label)

    result=fit_clf.predict(data)
    accuracy=float(np.sum(result==label))/len(label)
    print "Training accuracy is " + str(accuracy)

    scores = cross_val_score(clf, data, label, cv=5)
    print "Cross validation score is "+ str(scores.mean())

    return fit_clf

def get_palette():
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]

    return palette

def convert_label_to_trainid(current_layer_value):
    # convert label
    unique_values_in_array = np.unique(current_layer_value)
    unique_values_in_array = np.sort(unique_values_in_array)
    unique_values_in_array = unique_values_in_array[::-1]
    for unique_value in unique_values_in_array:
        converted_value = id2label[unique_value].trainId
        current_layer_value[current_layer_value == unique_value] = converted_value
    return  current_layer_value

def get_quality(superpixel_label,current_all_layer_values, index_superpixel):
    quality=False
    # ground-truth of current superpixel
    predict_label = superpixel_label[superpixel_label == index_superpixel]

    # superpixel too small
    if len(predict_label)<=20:
        return False,0,[]

    # if gt_label does not agree with each other uniformly in this area. (Set agreement threshold to 0.5 by default)
    agreement_threshold=0.5
    predict_label_count=Counter(predict_label).most_common()
    predict_label_consistency_rate=float(predict_label_count[0][1])/len(predict_label)

    if predict_label_consistency_rate<agreement_threshold:
        return False,predict_label_consistency_rate,predict_label_count

    # otherwise pass the quality test
    return True,predict_label_consistency_rate,predict_label_count


def get_feature_single_superpixel(superpixel_label,current_all_layer_values,index_superpixel,gt_label_consistency_rate,gt_label_count):
    binary_mask=(superpixel_label == index_superpixel).astype(np.uint8)

    # plt.imshow(binary_mask)
    # plt.show()

    feature=[]

    # feature dimension 0: label_consistency_rate
    feature.extend([gt_label_consistency_rate])

    # feature dimension 1: area
    contours, hierarchy = cv2.findContours(binary_mask, 0, 2)
    contours_with_holes, hierarchy_with_holes = cv2.findContours(binary_mask, 1, 2)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    feature.extend([area])

    # feature dimension 2: perimeter
    perimeter = cv2.arcLength(cnt, True)
    feature.extend([perimeter])

    # feature dimension 3,4: centroid
    M = cv2.moments(cnt)
    cx = int(M['m10'] / (M['m00']+1e-5))
    cy = int(M['m01'] / (M['m00']+1e-5))
    feature.extend([cx,cy])

    # feature dimension 5: convexity
    convexity=int(cv2.isContourConvex(cnt))
    feature.extend([convexity])

    # feature dimension 6,7,8,9,10: minAreaRectangle centerx, centery, width, height, angle of rotation
    rect = cv2.minAreaRect(cnt)
    feature.extend([rect[0][0],rect[0][1],rect[1][0],rect[1][1],rect[2]])

    # feature dimension  11,12,13: minEnclosingCircle centerx, centery, radius
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    feature.extend([int(x),int(y),radius])

    # feature dimension 14: aspect ratio
    x_rect,y_rect,w,h=cv2.boundingRect(cnt)
    aspect_ratio=float(w)/h
    feature.extend([aspect_ratio])

    # feature dimension 15: extent
    rect_area = w * h
    extent = float(area) / (rect_area+1e-5)
    feature.extend([extent])

    # feature dimension 16, 17: convex hull area and solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / (hull_area+1e-5)
    feature.extend([hull_area,solidity])

    # feature dimension 18: equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi)
    feature.extend([equi_diameter])

    # feature dimension 19, 20, 21: major axis length, minor axis length, orientation
    if cnt.shape[0]<=5:
        feature.extend([0, 0, 0])
    else:
        (x_ecclipse, y_ecclipse), (MA, ma), angle = cv2.fitEllipse(cnt)
        feature.extend([MA, ma, angle])

    # feature dimension 22: Euler number
    Euler_number=len(contours)-len(contours_with_holes)
    feature.extend([Euler_number])

    # feature dimension 23, 24, 25, 26, 27, 28, 29, 30: extreme points
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    feature.extend([leftmost[0], bottommost[1], rightmost[0], rightmost[1], topmost[0], topmost[1], bottommost[0], bottommost[1]])

    # feature dimension 31, 32, 33, 34, 35, 36, 37: Hu moments
    Hu_moments=cv2.HuMoments(M).flatten()
    feature.extend(Hu_moments)

    # feature dimension 38, 39, 40, 41, 42, 43: prediction of our models and consistency
    for layer_index in range(current_all_layer_values.shape[2]):
        current_layer_current_superpixel_label = current_all_layer_values[:,:,layer_index][superpixel_label == index_superpixel]
        current_layer_label_count = Counter(current_layer_current_superpixel_label).most_common()
        current_layer_consistency_rate = float(current_layer_label_count[0][1]) / len(current_layer_current_superpixel_label)
        feature.extend([current_layer_label_count[0][0],current_layer_consistency_rate])


    return feature

def predict(superpixel_data,gt_files,folder_files,classifier):
    # iterate through all images
    feature_set=[]
    label_set=[]
    num_features=0
    for index in range(len(superpixel_data)):
        print str(index) + ' ' + superpixel_data[index].split('/')[-1]
        current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
        current_gt = cv2.imread(gt_files[index], 0)

        # gather prediction maps, form multi-layer maps
        current_all_layer_values = np.zeros((1024, 2048, len(folder_files)))
        for key, value in folder.iteritems():
            current_layer_value = cv2.imread(folder_files[key][index], 0)
            current_all_layer_values[:, :, key - 1]=convert_label_to_trainid(current_layer_value)

        # iterate through superpixel data of current image
        superpixel_label = current_superpixel_data[1]
        num_superpixels = np.max(superpixel_label) + 1
        for index_superpixel in range(num_superpixels):
            # decide the quality of current superpixel. Note: this is actually a test phase, so there should not be a GT!
            quality,gt_label_consistency_rate,gt_label_count = get_quality(superpixel_label,current_all_layer_values, index_superpixel)

            if not quality:
                # why bother to predict those regions? set to ignore label.
                superpixel_label[superpixel_label==index_superpixel]=255
                continue

            # TODO: MODIFY FOR TEST IMAGES/SUPERPIXELS
            # extract a 40 dimensional feature for current super pixel
            feature=get_feature_single_superpixel(superpixel_label,current_all_layer_values, index_superpixel,gt_label_consistency_rate,gt_label_count)
            predicted_label = classifier.predict(feature)

    return feature_set, label_set



if __name__ == '__main__':
    gt_folder = '/home/panquwang/Dataset/CityScapes/gtFine/val/'
    gt_files=glob.glob(os.path.join(gt_folder,"*","*gtFine_labelTrainIds.png"))
    gt_files.sort()

    superpixel_result_folder='/mnt/scratch/panqu/SLIC/merged_results/2016_08_24_11:21:59/data/'
    superpixel_data=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data.sort()

    training_feature_location='/mnt/scratch/panqu/SLIC/'
    all_feature_data = cPickle.load(open(os.path.join(training_feature_location,'features.dat'), "rb"))


    # test lower bound
    test_lower_bound(all_feature_data)

    # random forest classifier
    classifier=random_forest_classifier(all_feature_data)


    # prediction for validation set
    # base feature map
    folder={}
    # base
    folder[1]='/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/val/val-epoch-35-CRF/score/'
    # truck, wall
    folder[2]='/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/val/val-epoch-39-CRF-050/score/'
    # bus, train
    folder[3]='/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/val/val-epoch-33-CRF/score/'
    folder_files={}
    previous_key=0
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()

    print "start to predict..."

    feature_set,label_set=predict(superpixel_data,gt_files,folder_files,classifier)



