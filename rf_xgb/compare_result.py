#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
================================
SLIC superpixel
--------------------------------
input
    |- $image/videofile
    |- $stepsize
    |- $M
output
=================================
ogaki@iis.u-tokyo.ac.jp
2012/09/07
'''
import cv2
import sys
import scipy
import scipy.linalg
import random
import math
import os.path
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import glob
import cPickle
from PIL import Image
from datetime import datetime
import collections
sys.path.append( os.path.normpath( os.path.join('/home/panquwang/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
from labels     import trainId2label,id2label

if __name__ == '__main__':
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]



    superpixel_result_folder='/mnt/scratch/panqu/SLIC/merged_results/2016_08_24_11:21:59/data/'
    gt_folder='/home/panquwang/Dataset/CityScapes/gtFine/val/'

    superpixel_data=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data.sort()

    gt_files=glob.glob(os.path.join(gt_folder,"*","*gtFine_labelTrainIds.png"))
    gt_files.sort()


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

    print "start to compare..."

    ignore_labels=[-1,255]




    # calculation of upper bound
    # iterate through all images
    total_accuracy=[]
    for index in range(len(superpixel_data)):
        print str(index)+' '+superpixel_data[index].split('/')[-1]
        current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
        current_gt=cv2.imread(gt_files[index],0)

        # gather prediction maps
        total_area=1024*2048
        current_all_layer_values=np.zeros((1024,2048,len(folder_files)))
        for key, value in folder.iteritems():
            current_layer_value = cv2.imread(folder_files[key][index], 0)
            # convert label
            unique_values_in_array = np.unique(current_layer_value)
            unique_values_in_array = np.sort(unique_values_in_array)
            unique_values_in_array = unique_values_in_array[::-1]
            for unique_value in unique_values_in_array:
                converted_value = id2label[unique_value].trainId
                current_layer_value[current_layer_value == unique_value] = converted_value
            current_all_layer_values[:,:,key-1] = current_layer_value


        # iterate through superpixel data of current image
        superpixel_label=current_superpixel_data[1]
        num_superpixels = np.max(superpixel_label)+1
        patch_consistency_scores=[]
        accuracy=0.0
        for index_superpixel in range(num_superpixels):
            # The prediction of superpixel across all channels
            # found_label_index=np.nonzero(superpixel_label==index_superpixel)
            current_all_layer_values_current_superpixel_label=current_all_layer_values[superpixel_label==index_superpixel]

            # ground-truth
            gt_label=current_gt[superpixel_label==index_superpixel]

            total_pixels_current_superpixel=len(gt_label)

            correct=0
            for single_pixel_index in range(len(gt_label)):
                if gt_label[single_pixel_index] in current_all_layer_values_current_superpixel_label[single_pixel_index]:
                    correct = correct + 1
                elif gt_label[single_pixel_index]==255 or gt_label[single_pixel_index]==-1:
                    correct = correct + 1

            accuracy=accuracy+float(correct)/total_area


            print "accuracy= " + str(accuracy)

        print "total accuracy for current image= " + str(accuracy)
        total_accuracy.append(accuracy)

    np.save('accuracy',total_accuracy)
    # compute upper bound
    averaged_total_score=np.average(total_accuracy)
    print "total averaged accuracy for current image= " + str(averaged_total_score)






