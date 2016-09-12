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
'''
import cv2
import sys
import scipy
import scipy.linalg
import random
import math
import os.path
import numpy as np
import matplotlib.pyplot as plt
import glob
import cPickle
from datetime import datetime
import fnmatch
import os
from pyspark import SparkContext, SparkConf

def argmin(_list):
    return _list.index(min(_list))

def gradient_img(colorsrc):
    '''
        http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    '''
    SCALE = 1
    DELTA = 0
    DDEPTH = cv2.CV_16S  ## to avoid overflow

    # grayscale image
    if len(colorsrc.shape)==2:
        graysrc = cv2.GaussianBlur(colorsrc, (3, 3), 0)

        ## gradient X ##
        gradx = cv2.Sobel(graysrc, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
        gradx = cv2.convertScaleAbs(gradx)

        ## gradient Y ##
        grady = cv2.Sobel(graysrc, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)
        grady = cv2.convertScaleAbs(grady)

        grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

        return grad

    # multi-channel image
    else:
        gradx_total = np.zeros((colorsrc.shape[0], colorsrc.shape[1]))
        grady_total = np.zeros((colorsrc.shape[0], colorsrc.shape[1]))
        for index in range(colorsrc.shape[2]):
            graysrc=colorsrc[:,:,index]
            graysrc = cv2.GaussianBlur(graysrc, (3, 3), 0)

            ## gradient X ##
            gradx = cv2.Sobel(graysrc, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
            gradx = cv2.convertScaleAbs(gradx)
            gradx_total=gradx_total+gradx

            ## gradient Y ##
            grady = cv2.Sobel(graysrc, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)
            grady = cv2.convertScaleAbs(grady)
            grady_total = grady_total + grady

            grad = cv2.addWeighted(gradx_total, 0.5, grady_total, 0.5, 0)

        return grad


class SlicCalculator(object):
    ## Superparameter
    M = 20  ##weight of color in distance
    INITCENTER_SEARCHDIFF = 1  ## minimum gradient
    ERROR_THRESHOLD = 10

    ## Config
    INIFINITY_DISTANCE = 1 << 23
    DEBUGFLAG = False

    def __init__(self, img, step=50, outfilename="SLICsuperpixel.img", stepsize=None, M=None, outparamfilename=None, result_dir='/mnt/scratch/panqu/SLIC/'):
        if not M is None:
            self.M = M
            self.MM = M * M
        if stepsize is None:
            stepsize = (int(step), int(step))
        if outparamfilename is None:
            base, ext = os.path.splitext(outfilename)
            outparamfilename = base + "_params.dat"
        self.stepsize = stepsize
        self.result_dir=result_dir
        self.filename = outfilename
        self.outparamfilename = outparamfilename
        self.img = img
        if self.DEBUGFLAG:
            cv2.imshow("test", self.img)
            cv2.waitKey(10)
        self.labimg = img
        # self.labimg = cv2.cvtColor(self.img, cv2.cv.CV_BGR2Lab)
        self.location_info=np.asarray([[[x, y] for y in xrange(self.labimg.shape[1])] for x in xrange(self.labimg.shape[0])])
        if len(self.labimg.shape)==2:
            self.labimg=np.expand_dims(self.labimg, axis=2)
        self.xylab = np.concatenate((self.location_info,self.labimg), axis=2)
        print("Init finished.")

    def _initialize_center_grid(self, cluster_size):
        xs = range(cluster_size[0] / 2, self.img.shape[0], cluster_size[0])
        ys = range(cluster_size[1] / 2, self.img.shape[1], cluster_size[1])
        return [scipy.array([x, y]) for x in xs for y in ys]

    def _getneighborhood(self, point2d, distanceset):
        '''
        Neighbor coordinates from points2d
        point2d: current center
        distanceset :pre-defined step size
        '''
        return scipy.array([[px, py]
                            for px in range(max(int(point2d[0]) - distanceset[0], 0),
                                            min(int(point2d[0]) + distanceset[0] + 1, self.img.shape[0]))
                            for py in range(max(int(point2d[1]) - distanceset[1], 0),
                                            min(int(point2d[1]) + distanceset[1] + 1, self.img.shape[1]))
                            ])

    def _getneighborhood_in_image(self, point2d, distanceset):
        '''
        distanceset: [distancex, distancey]
        '''
        points = self._getneighborhood(point2d, distanceset)
        result=np.zeros((len(points),len(points[0])+self.labimg.shape[2]))
        # labs = self.labimg[points[0][0]:points[-1][0] + 1, points[0][1]:points[-1][1] + 1] # get pixel value for this 3x3 neighborhood obtained from points

        for i in range(len(points)):
           result[i]=np.append(points[i],self.labimg[points[i][0],points[i][1]])

        return result

    def _search_minimum_gradient(self, point2d, distance):
        searchpoints = self._getneighborhood_in_image(point2d, [distance, distance])
        searchvals = [self.grad[point[0], point[1]] for point in searchpoints]
        return searchpoints[argmin(searchvals)][:2]

    def _initialize_center_avoidedge(self, centers, distance):
        self.grad = gradient_img(self.img)
        return [self._search_minimum_gradient(point, distance) for point in centers]

    def _initialize_center(self, cluster_size):
        # initialize centers based on cluster size provided
        centers = self._initialize_center_grid(cluster_size)
        # make sure all centers are not on an edge based on gradient map of a 3x3 region.
        centers = self._initialize_center_avoidedge(centers, self.INITCENTER_SEARCHDIFF)
        # append color information to all initialized center locationss.
        centers = [scipy.concatenate((center, self.labimg[center[0]][center[1]])) for center in centers]
        return centers

    def _initassignments(self):
        width, height = self.img.shape[:2]
        self.assignedindex = scipy.array([[0 for i in xrange(height)] for j in xrange(width)])
        self.assigneddistance = scipy.array([[self.INIFINITY_DISTANCE for i in xrange(height)] for j in xrange(width)])

    def calcdistance_mat(self, points, center, spatialmax):
        ## -- L2norm optimized -- ##
        center = scipy.array(center)

        location_center=center[:2]
        color_center=center[2:]

        location_points=points[:,:,:2]
        color_points=points[:,:,2:]

        difs_location=location_points-location_center
        difs_color=1-np.equal(color_points,color_center)
        if len(difs_color.shape)==2:
            difs_color=np.expand_dims(difs_color, axis=2)

        difs=np.concatenate((difs_location,difs_color),axis=2)

        norm = (difs ** 2).astype(float)
        norm[:, :, 0:2] *= (float(self.MM) / (spatialmax * spatialmax))  # color weight on location term
        norm = scipy.sum(norm, 2)
        return norm

    def assignment(self, centers, stepsize):
        stepmax = max(stepsize)
        for assignment_index, center in enumerate(centers):
            points = self._getneighborhood(center[:2], stepsize)

            searchpoints = self.xylab[points[0][0]:points[-1][0] + 1, points[0][1]:points[-1][1] + 1]
            searchassignedindex = self.assignedindex[points[0][0]:points[-1][0] + 1, points[0][1]:points[-1][1] + 1]
            searchassigneddistance = self.assigneddistance[points[0][0]:points[-1][0] + 1,
                                     points[0][1]:points[-1][1] + 1]

            distancemat = self.calcdistance_mat(searchpoints, center, stepmax)

            searchassignedindex[searchassigneddistance > distancemat] = assignment_index
            searchassigneddistance[searchassigneddistance > distancemat] = distancemat[
                searchassigneddistance > distancemat]

    def update(self, centers):
        # sums = [scipy.zeros(5) for i in range(len(centers))]
        # nums = [0 for i in range(len(centers))]
        # width, height = self.img.shape[:2]
        print "E step"
        new_centers=[]
        nan_record=[]

        for i in xrange(len(centers)):
            current_region=self.xylab[self.assignedindex == i]
            if current_region.size>0: #non-empty region
                new_centers.append(scipy.mean(current_region, 0))
            else: # empty region
                nan_record.append(i)

        # after we get full nan_record list, update assignment index (elimnate those indexes in reverse order)
        for nan_value in nan_record[::-1]:
            self.assignedindex[self.assignedindex>nan_value]=self.assignedindex[self.assignedindex>nan_value]-1


        for new_center_index in range(len(new_centers)):
            # print new_center_index
            new_centers[new_center_index][0] = math.floor(new_centers[new_center_index][0])
            new_centers[new_center_index][1] = math.floor(new_centers[new_center_index][1])
            new_centers[new_center_index][2:]=self.labimg[math.floor(new_centers[new_center_index][0])][math.floor(new_centers[new_center_index][1])]

        return new_centers,nan_record

    def calcerror(self, centers, prevcenters,nan_record):
        '''
        L2 norm of location
        '''
        for nan_index in nan_record[::-1]:
            del prevcenters[nan_index]

        # error=sum([scipy.dot(now[:2] - prev[:2], now[:2] - prev[:2]) + scipy.dot(1-np.equal(now[2:], prev[2:]), 1-np.equal(now[2:], prev[2:])) for now, prev in zip(centers, prevcenters)])
        error=sum([scipy.dot(1-np.equal(now[2:], prev[2:]), 1-np.equal(now[2:], prev[2:])) for now, prev in zip(centers, prevcenters)])

        print "error:", error
        return error

    def iteration(self, centers, stepsize):
        error = sum([scipy.dot(center[:2], center[:2]) for center in centers])
        while error > self.ERROR_THRESHOLD:
            self.assignment(centers, stepsize)  ## M-step Note step size is the initial length/width of a superpixel.
            prevcenters=centers
            centers,nan_record=self.update(centers)  ## E-step
            error = self.calcerror(centers, prevcenters,nan_record)
            print "L2 error:", error

            if self.DEBUGFLAG:
                base, ext = os.path.splitext(self.filename)
                self.filename = base.split("_error")[0] + "_error" + str(error) + ext
                self.resultimg(centers)
        return (centers, self.assignedindex)

    def resultimg(self, centers):
        print "show result"
        result = scipy.zeros(self.img.shape[:2], scipy.uint8)
        width, height = result.shape[:2]
        if len(result.shape)>2:
            color_channels=result.shape[2]
        else:
            color_channels=1
        colors = [scipy.array([int(random.uniform(0, 255)) for i in xrange(1)]) for j in xrange(len(centers))]
        for x in xrange(width):
            for y in xrange(height):
                result[x, y] = colors[self.assignedindex[x][y]]

        # cv2.imshow("result", result)
        # cv2.waitKey(10)
        cv2.imwrite(os.path.join(self.result_dir,self.filename+'_superpixel.png'), result)

    def saveparams(self, centers, filename=None):
        if filename is None:
            filename = self.outparamfilename
        cPickle.dump((centers, self.assignedindex), open(os.path.join(self.result_dir,filename), "w+"))

    def calc(self):
        centers = self._initialize_center(self.stepsize)
        self._initassignments() # assign every pixel to the same superpixel and initalize the distance term.
        centers, self.assignedindex=self.iteration(centers, self.stepsize)

        self.resultimg(centers)
        self.saveparams(centers)

def processing(index,total_files,folder_files,img_height,img_width,img_channels,result_dir,step):
    print "start file "+folder_files[1][index].split('/')[-1]
    # get initial multi-channel input (channel can be arbitrary positive integer)
    input=np.zeros((img_height,img_width,img_channels))
    for key,value in folder_files.iteritems():
        input[:,:,int(key)-1]=cv2.imread(folder_files[key][index],0)

    outfilename=folder_files[key][index].split('/')[-1][:-4]

    calculator = SlicCalculator(input, step=step, M=0, outfilename=outfilename,result_dir=result_dir)

    # calculator.DEBUGFLAG = True
    calculator.calc()

def spark_processing(i):
    dataset = 'train'
    # use 500 training subfolder
    folder = {}
    # base:
    folder[1] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_bigger_patch_epoch_35/', dataset,
                             dataset + '_sub-epoch-35-CRF')
    # scale 05
    folder[2] = os.path.join('/mnt/scratch/panqu/to_pengfei/asppp_cell2_epoch_39/', dataset,
                             dataset + '_sub-epoch-39-CRF-050')
    # wild atrous
    folder[3] = os.path.join('/mnt/scratch/pengfei/crf_results/yenet_asppp_wild_atrous_epoch16_crf_' + dataset + '_sub',
                             'score')
    # deconv
    folder[4] = os.path.join('/mnt/scratch/pengfei/crf_results/deeplab_deconv_epoch30_' + dataset + '_sub_crf', 'score')

    folder_files = {}
    previous_key = 0
    for key, value in folder.iteritems():
        folder_files[key] = glob.glob(os.path.join(value, '*.png'))
        folder_files[key].sort()
        if int(key) >= 2 and not len(folder_files[key]) == len(folder_files[previous_key]):
            raise ValueError('file folder lengths are not equal!')
        previous_key = key

    # Initialization
    # Step: initial length/width of one superpixel
    # M: color ratio
    img_width=2048
    img_height=1024
    img_channels=len(folder_files)
    num_superpixels = 50000
    step=int(math.ceil((img_width*img_height/num_superpixels)**0.5))
    result_dir=os.path.join('/mnt/scratch/panqu/SLIC/server_'+dataset)

    total_files=len(folder_files[1])

    processing(i, total_files, folder_files, img_height, img_width, img_channels, result_dir,step)

    return 1


num_cores=75
conf = SparkConf()
conf.setAppName("semantic_segmentation").setMaster("spark://192.168.1.132:7077")
conf.set("spark.scheduler.mode", "FAIR")
conf.set("spark.cores.max", num_cores)
sc = SparkContext(conf=conf)

range_i = range(0, 500)
RDDList = sc.parallelize(range_i, num_cores)
print '------------------------------------start spark-----------------------------------'

mapper = RDDList.map(spark_processing).reduce(lambda a, b : a+b)
print "total files processed {}".format(mapper)
print '-------------------------------------done-----------------------------------------'



