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
import sys
sys.path.insert(0,'/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
import cv2
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
from datetime import datetime
import collections
import copy
from PIL import Image
sys.path.append( os.path.normpath( os.path.join('/home/panquwang/.local/lib/python2.7/') ) )
from joblib import Parallel, delayed
import multiprocessing

def find_unique_labels(current_superpixel_data):
    # find unique vectors (number of dimensions equals number of layers)
    current_superpixel_data_channel_info_all = np.array([x[2:] for x in current_superpixel_data[0]])
    current_superpixel_data_channel_info_all_transformed = np.ascontiguousarray(
    current_superpixel_data_channel_info_all).view(np.dtype((np.void,
                                                                 current_superpixel_data_channel_info_all.dtype.itemsize *
                                                                 current_superpixel_data_channel_info_all.shape[1])))
    _, idx = np.unique(current_superpixel_data_channel_info_all_transformed, return_index=True)
    unique_labels = current_superpixel_data_channel_info_all[idx].tolist()

    return unique_labels

def assign_updated_labels(num_superpixels,current_superpixel_data,unique_labels):
    # iterate: assign updated labels for all superpixels
    return_map = np.ones((current_superpixel_data[1].shape[0], current_superpixel_data[1].shape[1])) * (-1)
    for superpixel_index in range(num_superpixels):
        # print superpixel_index
        current_superpixel_data_channel_info = current_superpixel_data[0][superpixel_index][2:].tolist()
        index_in_list = unique_labels.index(current_superpixel_data_channel_info)
        # replace labels
        return_map[current_superpixel_data[1] == superpixel_index] = index_in_list
    return return_map

def palette(new_superpixel_label):
    palette = [0] * 256 * 3
    for index in range(len(palette)):
        palette[index] = np.random.choice(range(0,255))
    return palette

def parallel_processing(index,superpixel_data,superpixel_images,place_to_save):
    print index
    current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
    to_be_saved_file_name=superpixel_data[index].split('/')[-1][:-11]+'_merged.png'
    to_be_saved_data_name = superpixel_data[index].split('/')[-1][:-11] + '_merged.dat'

    # statisticst
    num_superpixels=len(current_superpixel_data[0])
    superpixel_labels=current_superpixel_data[1]

    # find unique labels
    unique_labels=find_unique_labels(current_superpixel_data)

    # iterate: assign updated labels for all superpixels
    return_map=assign_updated_labels(num_superpixels,current_superpixel_data,unique_labels)

    # plt.imshow(current_superpixel_data[1])
    # plt.show()


    print "Finding connected component for "+str(index)+' '+superpixel_data[index].split('/')[-1]

    new_superpixel_label=0
    chosen_label_values=[]
    final_map=np.ones((1024,2048))*(num_superpixels+10)
    for index_unique_label,unique_label in enumerate(unique_labels):
        current_unique_label_layer=return_map==index_unique_label
        current_unique_label_layer=current_unique_label_layer.astype(np.uint8)

        current_unique_label_layer_connected_component=cv2.connectedComponents(current_unique_label_layer, connectivity=8)
        total_connected_components=current_unique_label_layer_connected_component[0]
        # plt.imshow(current_unique_label_layer_connected_component[1])
        # plt.show()
        for index_connected_component in range(1,total_connected_components):
            # chosen_label_value=np.random.choice(label_array)
            final_map[current_unique_label_layer_connected_component[1]==index_connected_component]=new_superpixel_label
            # label_array.remove(chosen_label_value)
            chosen_label_values.append(new_superpixel_label)
            new_superpixel_label=new_superpixel_label+1


    # render
    my_palette=palette(new_superpixel_label)
    result_img = Image.fromarray(final_map.astype(np.uint8)).convert('P')
    result_img.putpalette(my_palette)
    result_img.save(os.path.join(place_to_save, 'visualization', to_be_saved_file_name))


    # save data
    final_superpixel_data=current_superpixel_data+(final_map,unique_labels,chosen_label_values)
    cPickle.dump(final_superpixel_data, open(os.path.join(place_to_save, 'data', to_be_saved_data_name), "w+"))





if __name__ == '__main__':
    dataset='train'

    # superpixel_result_folder='/mnt/scratch/panqu/SLIC/server_combine_all_'+dataset+'/'
    superpixel_result_folder='/mnt/scratch/panqu/SLIC/server_train/2016_09_12_15:33:44'
    original_files_folder='/home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'/'
    gt_folder='/home/panquwang/Dataset/CityScapes/gtFine/'+dataset+'/'

    superpixel_images=glob.glob(os.path.join(superpixel_result_folder,'*.png'))
    superpixel_images.sort()

    superpixel_data=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data.sort()

    # original_files=glob.glob(os.path.join(original_files_folder,"*","*.png"))
    # original_files.sort()
    #
    # gt_files=glob.glob(os.path.join(gt_folder,"*","*gtFine_labelTrainIds.png"))
    # gt_files.sort()

    time=datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    place_to_save = '/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_'+dataset+'_subset/'
    if not os.path.exists(place_to_save):
        os.makedirs(place_to_save)
        os.makedirs(os.path.join(place_to_save, 'data'))
        os.makedirs(os.path.join(place_to_save, 'visualization'))


    num_cores = multiprocessing.cpu_count()
    range_i=range(0,500)

    Parallel(n_jobs=num_cores)(delayed(parallel_processing)(i,superpixel_data,superpixel_images,place_to_save) for i in range_i)







