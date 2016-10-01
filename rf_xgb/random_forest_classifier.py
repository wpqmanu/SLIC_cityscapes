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
sys.path.append( os.path.normpath( os.path.join('/home/panquwang/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
import labels
from labels     import trainId2label,id2label
# matplotlib.use('Qt4Agg')
from feature_extraction import get_feature_single_superpixel
import feature_extraction_with_neighbor

def random_forest_classifier(all_feature_data):
    input_data=np.asarray(all_feature_data[0])
    label=np.asarray(all_feature_data[1])

    data=input_data[:,:]
    # data=sklearn.preprocessing.normalize(data,axis=0)

    clf = RandomForestClassifier(n_estimators=1,
                                 verbose=True,
                                 n_jobs=8,
                                 max_features=3,
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=2,
                                 max_leaf_nodes=None
                                 )
    fit_clf=clf.fit(data,label)

    # result=fit_clf.predict(data)
    # accuracy=float(np.sum(result==label))/len(label)
    # print "Training accuracy is " + str(accuracy)

    # scores = cross_val_score(clf, data, label, cv=10)
    # print "Cross validation score is "+ str(scores.mean())

    return fit_clf

def get_palette():
    # get palette
    trainId2colors = {label.trainId: label.color for label in labels.labels}
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
    for unique_value in unique_values_in_array:
        converted_value = id2label[unique_value].trainId
        current_layer_value[current_layer_value == unique_value] = converted_value
    return  current_layer_value

def convert_trainid_to_label(label):
    unique_values_in_final_array = np.unique(label)
    unique_values_in_final_array = np.sort(unique_values_in_final_array)
    unique_values_in_final_array = unique_values_in_final_array[::-1]
    for unique_value in unique_values_in_final_array:
        if unique_value < 19:
            converted_value = trainId2label[unique_value].id
        else:
            converted_value = 255
        label[label == unique_value] = converted_value
    label[label == 255] = 0
    return label

def get_quality(superpixel_label,current_all_layer_values, index_superpixel):
    quality=False
    # ground-truth of current superpixel
    predict_label = superpixel_label[superpixel_label == index_superpixel]

    # superpixel too small
    if len(predict_label)<=10:
        return False,0,[]

    # if prediction label does not agree with each other uniformly in this area. (Set agreement threshold to 0.5 by default)
    agreement_threshold=0.5
    predict_label_count=Counter(predict_label).most_common()
    predict_label_consistency_rate=float(predict_label_count[0][1])/len(predict_label)

    if predict_label_consistency_rate<agreement_threshold:
        return False,predict_label_consistency_rate,predict_label_count

    # otherwise pass the quality test
    return True,predict_label_consistency_rate,predict_label_count

def test_lower_bound(all_feature_data,superpixel_data,gt_files,folder_files,original_image_files,result_location,is_test_lower_bound):
    data=np.asarray(all_feature_data[0])
    label=np.asarray(all_feature_data[1])

    # majority voting for classifier
    data_predictions=data[:,[38,40,42]]
    data_predictions_output=[]
    for temp_index in range(data_predictions.shape[0]):
        value=list(data_predictions[temp_index])
        data_predictions_output.append(max(set(value), key=value.count))

    lower_bound = float(np.sum(np.asarray(data_predictions_output) == label)) / len(label)
    print "Accuracy of lower bound is " + str(lower_bound)

    predict(superpixel_data, gt_files, folder_files, classifier, original_image_files, result_location,is_test_lower_bound)

def get_only_3_dim_data(all_feature_data_train,all_feature_data_val):
    all_feature_data_train_data = all_feature_data_train[0]
    all_feature_data_val_data = all_feature_data_val[0]

    all_feature_data_train_data_label_only = []
    all_feature_data_val_data_label_only = []

    label_list = [38, 40, 42]
    for feature_row in all_feature_data_train_data:
        temp_list = []
        for label_in_list in label_list:
            temp_list.append(feature_row[label_in_list])
        all_feature_data_train_data_label_only.append(temp_list)

    for feature_row in all_feature_data_val_data:
        temp_list = []
        for label_in_list in label_list:
            temp_list.append(feature_row[label_in_list])
        all_feature_data_val_data_label_only.append(temp_list)

    saved_location = '/mnt/scratch/panqu/SLIC/features/'
    cPickle.dump((all_feature_data_train_data_label_only, all_feature_data_train[1]),
                 open(os.path.join(saved_location, 'features_' + 'train' + '_3.dat'), "w+"))
    cPickle.dump((all_feature_data_val_data_label_only, all_feature_data_val[1]),
                 open(os.path.join(saved_location, 'features_' + 'val' + '_3.dat'), "w+"))


def predict(superpixel_data,gt_files,folder_files,classifier,original_image_files,result_location,is_test_lower_bound,is_use_neighbor):

    img_width=2048
    img_height=1024

    # iterate through all images
    for index in range(len(superpixel_data)):
        file_name = superpixel_data[index].split('/')[-1][:-4]+'.png'
        print str(index) + ' ' + file_name
        current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
        current_gt = cv2.imread(gt_files[index], 0)
        original_image = cv2.imread(original_image_files[index])

        # gather prediction maps, form multi-layer maps
        current_all_layer_values = np.zeros((img_height, img_width, len(folder_files)))
        for key, value in folder.iteritems():
            current_layer_value = cv2.imread(folder_files[key][index], 0)
            current_all_layer_values[:, :, key - 1]=convert_label_to_trainid(current_layer_value)




        # iterate through superpixel data of current image
        superpixel_label = current_superpixel_data[2]
        num_superpixels = np.max(superpixel_label) + 1

        # statistics
        each_label_size = []
        for index_superpixel in range(int(num_superpixels)):
            length = len(superpixel_label[superpixel_label == index_superpixel])
            each_label_size.append(length)


        final_map=np.ones((img_height, img_width))*(int(num_superpixels)+10)
        superpixel_feature_set=[]
        superpixel_index_set=[]
        superpixel_categorical_label=[]
        for index_superpixel in range(int(num_superpixels)):
            # decide the quality of current superpixel. Note: this is actually a test phase, so there should not be a GT!
            quality,gt_label_consistency_rate,predict_label_count = get_quality(superpixel_label,current_all_layer_values, index_superpixel)

            if not quality:
                # why bother to predict those regions? set to ignore label.
                # final_map[superpixel_label==index_superpixel]=255
                continue

            #MODIFY FOR TEST IMAGES/SUPERPIXELS

            if not is_use_neighbor:
                # extract a 100 dimensional feature for current super pixel
                feature, label, categorical_label=get_feature_single_superpixel(superpixel_label,current_all_layer_values, index_superpixel,gt_label_consistency_rate,predict_label_count)
            else:
                # extract a 240 dimensional feature for current super pixel
                feature, label, categorical_label=feature_extraction_with_neighbor.get_feature_single_superpixel(superpixel_label,current_all_layer_values, index_superpixel,gt_label_consistency_rate,predict_label_count,each_label_size)


            # save to a single set to increase processing speed
            feature_get=[feature[i] for i in [38,40,42]]
            superpixel_feature_set.append(feature_get)
            superpixel_index_set.append(index_superpixel)
            superpixel_categorical_label.append(categorical_label)

            if is_test_lower_bound:
                label_candidates=categorical_label
                label_selected = max(set(label_candidates), key=label_candidates.count)
                final_map[superpixel_label == index_superpixel] = label_selected

            # else:
            #     predicted_label_probs = classifier.predict_proba(feature)
            #     label_candidates_probs=[predicted_label_probs[0][i] for i in label_candidates]
            #     label_selected=label_candidates[np.argmax(label_candidates_probs)]
            # final_map[superpixel_label == index_superpixel] = label_selected

        # to improve speed, predict the features for one pass.
        predicted_label_probs = classifier.predict_proba(superpixel_feature_set)
        # then assign label to all superpixel
        for predicted_index,superpixel_index in enumerate(superpixel_index_set):
            predicted_current_superpixel_label_probs=predicted_label_probs[predicted_index]
            label_candidates = superpixel_categorical_label[predicted_index]
            label_candidates_probs = [predicted_current_superpixel_label_probs[i] for i in label_candidates]
            label_selected = label_candidates[np.argmax(label_candidates_probs)]
            final_map[superpixel_label == superpixel_index] = label_selected
        final_map[final_map>int(num_superpixels)]=255

        # save score
        final_map_saved=copy.deepcopy(final_map)
        score=convert_trainid_to_label(final_map)
        cv2.imwrite(os.path.join(result_location,'score',file_name),score)

        # save visualization
        # original image
        concat_img = Image.new('RGB', (img_width * 3, img_height))
        concat_img.paste(Image.fromarray(original_image[:, :, [2, 1, 0]]).convert('RGB'), (0, 0))
        # ground truth
        gt_img = Image.open(gt_files[index])
        concat_img.paste(gt_img, (img_width, 0))
        # prediction
        final_map_saved = final_map_saved.astype(np.uint8)
        result_img = Image.fromarray(final_map_saved).convert('P')
        palette = get_palette()
        result_img.putpalette(palette)
        # concat_img.paste(result_img, (2048*2,0))
        concat_img.paste(result_img.convert('RGB'), (img_width * 2, 0))
        concat_img.save(os.path.join(result_location, 'visualization', file_name))



if __name__ == '__main__':

    dataset='val'
    is_test_lower_bound=0
    is_use_neighbor=1

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
    # feature1 = '/mnt/scratch/panqu/SLIC/features/features_train_100_1.dat'
    # feature2 = '/mnt/scratch/panqu/SLIC/features/features_train_100_2.dat'
    # feature3 = '/mnt/scratch/panqu/SLIC/features/features_train_100_3.dat'
    # feature1_data = cPickle.load(open(feature1, "rb"))
    # feature2_data = cPickle.load(open(feature2, "rb"))
    # feature3_data = cPickle.load(open(feature3, "rb"))
    # all_features_data = (feature1_data[0]+feature2_data[0]+feature3_data[0],feature1_data[1]+feature2_data[1]+feature3_data[1])
    # cPickle.dump(all_features_data, open(os.path.join('/mnt/scratch/panqu/SLIC/features/', 'features_train_100.dat'), "w+"))
    if is_use_neighbor != 1:
        all_feature_data_train = cPickle.load(open(os.path.join(training_feature_location,'features','features_train_100.dat'), "rb"))
        all_feature_data_val = cPickle.load(open(os.path.join(training_feature_location, 'features', 'features_val_100.dat'), "rb"))
    else:
        all_feature_data_train = cPickle.load(open(os.path.join(training_feature_location, 'features', 'features_train_with_neighbor.dat'), "rb"))
        all_feature_data_val = cPickle.load(open(os.path.join(training_feature_location, 'features', 'features_val_with_neighbor.dat'), "rb"))


    # get_only_3_dim_data(all_feature_data_train,all_feature_data_val)


    result_location=os.path.join('/mnt/scratch/panqu/SLIC/prediction_result/', datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
    if not os.path.exists(result_location):
        os.makedirs(result_location)
        os.makedirs(os.path.join(result_location,'score'))
        os.makedirs(os.path.join(result_location,'visualization'))



    # prediction for validation set
    folder={}
    # base
    folder[1]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/',dataset, dataset+'-epoch-35-CRF', 'score')
    # truck, wall
    folder[2]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/',dataset, dataset+'-epoch-39-CRF-050', 'score')
    # bus, train
    folder[3]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/', dataset, dataset+'-epoch-33-CRF', 'score')

    folder_files={}
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()

    print "start to predict..."

    # # test lower bound
    if is_test_lower_bound:
        test_lower_bound(all_feature_data_train,superpixel_data,gt_files,folder_files,original_image_files,result_location,is_test_lower_bound)

    else:
        # random forest classifier
        classifier = random_forest_classifier(all_feature_data_train)
        predict(superpixel_data,gt_files,folder_files,classifier,original_image_files,result_location,is_test_lower_bound,is_use_neighbor)



