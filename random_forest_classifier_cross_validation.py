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
import copy
from collections import Counter
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
# sys.path.insert(0,'/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
import cv2

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


def cross_validation_random_forest_classifier(all_feature_data):
    data=np.asarray(all_feature_data[0])
    label=np.asarray(all_feature_data[1])

    #
    # criterion_all=['gini','entropy']
    # max_depth_all=[None,5]
    # min_samples_split_all=[1,2]
    criterion_all=['entropy']
    max_depth_all=[5]
    min_samples_split_all=[2]
    n_estimators_all = [25, 50, 75, 125, 175, 225, 325, 525, 1000]
    min_samples_leaf_all=[1,2]
    max_leaf_nodes_all=[None,10]
    max_features_all = [1, 5, 10, 15, 20, 25, 30, 35, None]


    for criterion in criterion_all:
        for max_depth in max_depth_all:
            for min_samples_split in min_samples_split_all:
                for n_estimators in n_estimators_all:
                    for min_samples_leaf in min_samples_leaf_all:
                        for max_leaf_nodes in max_leaf_nodes_all:
                            for max_features in max_features_all:
                                RF_params = {}
                                RF_params['criterion'] = criterion
                                RF_params['max_depth'] = max_depth
                                RF_params['min_samples_split'] = min_samples_split
                                RF_params['n_estimators'] = n_estimators
                                RF_params['min_samples_leaf'] = min_samples_leaf
                                RF_params['max_leaf_nodes'] = max_leaf_nodes
                                RF_params['max_features'] = max_features



                                clf = RandomForestClassifier(criterion=criterion,
                                                             max_depth=max_depth,
                                                             min_samples_split=min_samples_split,
                                                             n_estimators=n_estimators,
                                                             min_samples_leaf=min_samples_leaf,
                                                             max_leaf_nodes=max_leaf_nodes,
                                                             max_features=max_features,
                                                             n_jobs=-1,
                                                             verbose=True)
                                fit_clf=clf.fit(data,label)

                                result=fit_clf.predict(data)
                                accuracy=float(np.sum(result==label))/len(label)
                                print "Training accuracy is " + str(accuracy)

                                scores = cross_val_score(clf, data, label, cv=5)
                                print "Cross validation score is "+ str(scores.mean())

                                SLIC_result_file = '/mnt/scratch/panqu/SLIC/SLIC_result_file.txt'
                                with open(SLIC_result_file, "a") as SLIC_result_file:
                                    SLIC_result_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(
                                        str(criterion),
                                        str(max_depth),
                                        str(min_samples_split),
                                        str(n_estimators),
                                        str(min_samples_leaf),
                                        str(max_leaf_nodes),
                                        str(max_features),
                                        str(accuracy),
                                        str(scores.mean())))

if __name__ == '__main__':

    dataset='val'

    original_image_folder = '/home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'/'
    original_image_files=glob.glob(os.path.join(original_image_folder,"*","*.png"))
    original_image_files.sort()

    gt_folder = '/home/panquwang/Dataset/CityScapes/gtFine/'+dataset+'/'
    gt_files=glob.glob(os.path.join(gt_folder,"*","*gtFine_color.png"))
    gt_files.sort()

    superpixel_result_folder='/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_'+dataset+'/data/'
    superpixel_data=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data.sort()


    training_feature_location='/mnt/scratch/panqu/SLIC/'
    all_feature_data = cPickle.load(open(os.path.join(training_feature_location,'features','features_train_40.dat'), "rb"))

    # result_location=os.path.join('/mnt/scratch/panqu/SLIC/prediction_result/', datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
    # if not os.path.exists(result_location):
    #     os.makedirs(result_location)
    #     os.makedirs(os.path.join(result_location,'score'))
    #     os.makedirs(os.path.join(result_location,'visualization'))

    # # test lower bound
    # test_lower_bound(all_feature_data)

    # random forest classifier
    classifier=cross_validation_random_forest_classifier(all_feature_data)


