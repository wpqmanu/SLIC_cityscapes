import sys
import os.path
import numpy as np
import glob
import cPickle
from PIL import Image
import copy
from collections import Counter
# from feature_extraction import get_feature_single_superpixel
import cv2
# sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
# import labels
# from labels     import trainId2label,id2label
from joblib import Parallel, delayed
import multiprocessing

def get_palette():
    sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
    import labels
    from labels     import trainId2label,id2label
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
    sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
    import labels
    from labels     import trainId2label,id2label
    # convert label
    unique_values_in_array = np.unique(current_layer_value)
    unique_values_in_array = np.sort(unique_values_in_array)
    for unique_value in unique_values_in_array:
        converted_value = id2label[unique_value].trainId
        current_layer_value[current_layer_value == unique_value] = converted_value
    return  current_layer_value

def convert_trainid_to_label(label):
    sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
    import labels
    from labels     import trainId2label,id2label
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

def parallel_processing(rule_index, all_possible_rule_list, random_list,superpixel_data,gt_files,folder_files,original_image_files,is_test_lower_bound,is_use_neighbor,traverse_category_list):
    sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/SLIC_cityscapes/' ) ) )

    current_rule=all_possible_rule_list[rule_index]
    result_location = os.path.join('/mnt/scratch/panqu/SLIC/prediction_result/four_layers_rule_traverse_category_set_2345_after_trimming/', dataset,
                                   str(current_rule[0][0])+'_'+str(current_rule[0][1])+'_'+
                                   str(current_rule[0][2])+'_'+str(current_rule[0][3])+'_'+str(current_rule[1]))
    if not os.path.exists(result_location):
        os.makedirs(result_location)
        os.makedirs(os.path.join(result_location, 'score'))
        os.makedirs(os.path.join(result_location, 'visualization'))

    img_width=2048
    img_height=1024



    # iterate through all images
    # for index in range(len(superpixel_data)):
    for index in random_list:
        file_name = superpixel_data[index].split('/')[-1][:-4]+'.png'
        print str(index) + ' ' + file_name
        current_superpixel_data = cPickle.load(open(superpixel_data[index], "rb"))
        original_image = cv2.imread(original_image_files[index])

        # gather prediction maps, form multi-layer maps
        current_all_layer_values = np.zeros((img_height, img_width, len(folder_files)))
        for key, value in folder_files.iteritems():
            current_layer_value = cv2.imread(folder_files[key][index], 0)
            current_all_layer_values[:, :, key - 1]=convert_label_to_trainid(current_layer_value)

        # iterate through superpixel data of current image
        superpixel_label = current_superpixel_data[2]
        num_superpixels = np.max(superpixel_label) + 1

        # # statistics
        # each_label_size = []
        # for index_superpixel in range(int(num_superpixels)):
        #     length = len(superpixel_label[superpixel_label == index_superpixel])
        #     each_label_size.append(length)


        final_map=np.ones((img_height, img_width))*(int(num_superpixels)+10)
        superpixel_index_set=[]
        superpixel_categorical_label=[]
        for index_superpixel in range(int(num_superpixels)):
            # TODO: categorical label
            categorical_label=[]
            for layer_index in range(current_all_layer_values.shape[2]):
                current_layer_current_superpixel_label = current_all_layer_values[:,:,layer_index][superpixel_label == index_superpixel]
                current_layer_label_count = Counter(current_layer_current_superpixel_label).most_common()
                # current_layer_consistency_rate = float(current_layer_label_count[0][1]) / len(current_layer_current_superpixel_label)
                categorical_label.append(current_layer_label_count[0][0])

            superpixel_index_set.append(index_superpixel)
            superpixel_categorical_label.append(categorical_label)

        superpixel_categorical_label_copy=copy.deepcopy(superpixel_categorical_label)
        # assign label to all superpixels
        for predicted_index,superpixel_index in enumerate(superpixel_index_set):
            current_categorical_labels=superpixel_categorical_label_copy[predicted_index]
            # given a superpixel, set all except 4 big object labels to ignore label (255) to match input
            for current_categorical_label_index, current_categorical_label in enumerate(current_categorical_labels):
                if current_categorical_label not in traverse_category_list[:-1]:
                    current_categorical_labels[current_categorical_label_index]=traverse_category_list[-1]

            # apply the hard-coded primming rule to avoid bug such as (1,2,3,4) not treated as (255,2,3,4)
            if current_categorical_labels[0]==3:
                current_categorical_labels[0]=255
            if current_categorical_labels[1]==5 or current_categorical_labels[1]==2:
                current_categorical_labels[1]=255
            if current_categorical_labels[2]==5:
                current_categorical_labels[2]=255
            if current_categorical_labels[3]==3:
                current_categorical_labels[3]=255

            # if current superpixel meets the rule
            if current_categorical_labels==current_rule[0]:
                if current_rule[1]!=255: # if this label belongs to the 4 big object categories
                    final_map[superpixel_label == superpixel_index] = current_rule[1]
                else: # if this label belongs to other categories, get the label of that category (choose first if duplicate)
                    index_255=current_rule[0].index(255)
                    final_map[superpixel_label == superpixel_index] = superpixel_categorical_label[predicted_index][index_255]
            # if current superpixel does not meet the rule
            else:
                final_map[superpixel_label == superpixel_index] = superpixel_categorical_label[predicted_index][0]


        final_map[final_map>int(num_superpixels)]=255

        # save score
        # final_map_saved=copy.deepcopy(final_map)
        score=convert_trainid_to_label(final_map)
        cv2.imwrite(os.path.join(result_location,'score',file_name),score)

        # # save visualization
        # # original image
        # concat_img = Image.new('RGB', (img_width * 3, img_height))
        # concat_img.paste(Image.fromarray(original_image[:, :, [2, 1, 0]]).convert('RGB'), (0, 0))
        # # ground truth
        # gt_img = Image.open(gt_files[index])
        # concat_img.paste(gt_img, (img_width, 0))
        # # prediction
        # final_map_saved = final_map_saved.astype(np.uint8)
        # result_img = Image.fromarray(final_map_saved).convert('P')
        # palette = get_palette()
        # result_img.putpalette(palette)
        # # concat_img.paste(result_img, (2048*2,0))
        # concat_img.paste(result_img.convert('RGB'), (img_width * 2, 0))
        # concat_img.save(os.path.join(result_location, 'visualization', file_name))


if __name__ == '__main__':
    sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/SLIC_cityscapes/' ) ) )
    from feature_extraction import get_feature_single_superpixel
    sys.path.append(os.path.normpath(os.path.join('/mnt/scratch/panqu/Dataset/CityScapes/cityscapesScripts/scripts/', 'helpers' ) ) )
    import labels
    from labels     import trainId2label,id2label

    dataset='val'

    is_test_lower_bound=0
    is_use_neighbor=0
    is_get_subset_category_data=0
    is_calculate_purity=0


    original_image_folder = '/mnt/scratch/panqu/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'+dataset+'_for_traverse_frankfurt/'
    original_image_files=glob.glob(os.path.join(original_image_folder,"*.png"))
    original_image_files.sort()

    gt_folder = '/mnt/scratch/panqu/Dataset/CityScapes/gtFine/'+dataset+'_for_traverse_frankfurt/'
    gt_files=glob.glob(os.path.join(gt_folder,"*gtFine_color.png"))
    gt_files.sort()

    superpixel_result_folder='/mnt/scratch/panqu/SLIC/server_combine_all_merged_results_'+dataset+'_subset_frankfurt/data/'
    superpixel_data=glob.glob(os.path.join(superpixel_result_folder,'*.dat'))
    superpixel_data.sort()


    # prediction for validation set
    # folder={}
    # # base
    # folder[1]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/',dataset, dataset+'-epoch-35-CRF', 'score')
    # # truck, wall
    # folder[2]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/',dataset, dataset+'-epoch-39-CRF-050', 'score')
    # # bus, train
    # folder[3]=os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_atrous16_epoch_33/', dataset, dataset+'-epoch-33-CRF', 'score')

    folder = {}
    # base:
    folder[1] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/', dataset,
                             dataset + '-epoch-35-CRF_for_traverse_frankfurt')
    # scale 05
    folder[2] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/', dataset,
                             dataset + '-epoch-39-CRF-050_for_traverse_frankfurt')
    # wild atrous
    folder[3] = os.path.join(
        '/mnt/scratch/pengfei/crf_results/yenet_asppp_wild_atrous_epoch16_' + dataset + '_subset_crf_frankfurt')
    # deconv
    folder[4] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_deconv_epoch30_' + dataset + '_subset_crf_frankfurt')

    folder_files={}
    for key,value in folder.iteritems():
        folder_files[key]=glob.glob(os.path.join(value,'*.png'))
        folder_files[key].sort()

    print "start to predict..."

    traverse_list_length=4 # you have three layers for ensemble
    traverse_category_list=[2,3,4,5,255] # you only want to explore several categories (255 means all others)
    random_list=range(0,267)

    # enumerate all rules
    all_possible_rule_list=[]
    all_possible_rule_list.append(([2, 3, 3, 2], 3))
    all_possible_rule_list.append(([2, 3, 3, 255], 3))
    all_possible_rule_list.append(([4, 3, 3, 255], 3))
    all_possible_rule_list.append(([255, 4, 3, 4], 4))
    all_possible_rule_list.append(([255, 4, 4, 4], 4))
    all_possible_rule_list.append(([2, 255, 3, 2], 3))
    all_possible_rule_list.append(([2, 255, 3, 255], 3))
    all_possible_rule_list.append(([2, 255, 255, 255], 255))
    all_possible_rule_list.append(([255, 3, 3, 255], 3))
    all_possible_rule_list.append(([2, 3, 2, 2], 3))
    all_possible_rule_list.append(([255, 4, 4, 255], 4))
    all_possible_rule_list.append(([255, 255, 2, 2], 2))
    all_possible_rule_list.append(([255, 255, 4, 4], 4))
    all_possible_rule_list.append(([255, 3, 255, 255], 3))
    all_possible_rule_list.append(([4, 4, 3, 255], 3))
    all_possible_rule_list.append(([2, 3, 255, 255], 3))
    all_possible_rule_list.append(([4, 255, 2, 2], 255))
    all_possible_rule_list.append(([4, 255, 2, 4], 255))
    all_possible_rule_list.append(([2, 3, 4, 4], 4))
    all_possible_rule_list.append(([2, 3, 255, 2], 3))
    all_possible_rule_list.append(([2, 255, 4, 4], 4))
    all_possible_rule_list.append(([4, 3, 3, 4], 3))
    all_possible_rule_list.append(([4, 3, 4, 255], 255))
    all_possible_rule_list.append(([4, 3, 255, 255], 3))
    all_possible_rule_list.append(([4, 4, 4, 2], 2))
    all_possible_rule_list.append(([4, 4, 255, 255], 255))
    all_possible_rule_list.append(([4, 255, 2, 255], 255))
    all_possible_rule_list.append(([4, 255, 3, 255], 3))
    all_possible_rule_list.append(([4, 255, 4, 5], 255))
    all_possible_rule_list.append(([255, 3, 3, 2], 3))
    all_possible_rule_list.append(([255, 3, 255, 4], 4))
    all_possible_rule_list.append(([255, 4, 255, 4], 4))

    all_possible_rule_list.append(([2, 3, 255, 4], 3))
    all_possible_rule_list.append(([2, 4, 255, 4], 255))
    all_possible_rule_list.append(([2, 255, 255, 4], 255))
    all_possible_rule_list.append(([2, 255, 3, 5], 255))
    all_possible_rule_list.append(([2, 3, 255, 5], 255))
    all_possible_rule_list.append(([4, 255, 3, 5], 255))
    all_possible_rule_list.append(([4, 3, 255, 5], 5))
    all_possible_rule_list.append(([4, 255, 255, 2], 255))
    all_possible_rule_list.append(([4, 3, 255, 2], 255))
    all_possible_rule_list.append(([5, 255, 3, 4], 255))
    all_possible_rule_list.append(([5, 255, 4, 255], 255))
    all_possible_rule_list.append(([5, 255, 3, 5], 255))
    all_possible_rule_list.append(([5, 3, 255, 2], 255))
    all_possible_rule_list.append(([5, 3, 4, 255], 255))
    all_possible_rule_list.append(([5, 3, 2, 2], 3))
    all_possible_rule_list.append(([255, 4, 3, 2], 3))
    all_possible_rule_list.append(([255, 3, 2, 255], 3))
    all_possible_rule_list.append(([255, 3, 4, 4], 3))
    all_possible_rule_list.append(([4, 4, 255, 4], 255))




    len_rules=len(all_possible_rule_list) #565 rules in total


    num_cores = multiprocessing.cpu_count()
    range_i=range(0,51)
    Parallel(n_jobs=num_cores)(delayed(parallel_processing)(i,all_possible_rule_list,random_list,superpixel_data,gt_files,folder_files,original_image_files,is_test_lower_bound,is_use_neighbor,traverse_category_list) for i in range_i)




