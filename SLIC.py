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


def argmin(_list):
    return _list.index(min(_list))


def L2norm(vec):
    return sum([item * item for item in vec])


def L2norm_2d(vec):
    return vec[0] * vec[0] + vec[1] * vec[1]


def L2norm_3d(vec):
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]


def norm2d(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])


def norm3d(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def gradient_img(colorsrc):
    '''
        http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    '''
    SCALE = 1
    DELTA = 0
    DDEPTH = cv2.CV_16S  ## to avoid overflow

    graysrc = cv2.cvtColor(colorsrc, cv2.cv.CV_BGR2GRAY)
    graysrc = cv2.GaussianBlur(graysrc, (3, 3), 0)

    ## gradient X ##
    gradx = cv2.Sobel(graysrc, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
    gradx = cv2.convertScaleAbs(gradx)

    ## gradient Y ##
    grady = cv2.Sobel(graysrc, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)
    grady = cv2.convertScaleAbs(grady)

    grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

    return grad


class SlicCalculator(object):
    ## Superparameter
    M = 20  ##weight of color in distance
    INITCENTER_SEARCHDIFF = 1  ## minimum gradient
    ERROR_THRESHOLD = 50

    ## Config
    INIFINITY_DISTANCE = 1 << 23
    DEBUGFLAG = False

    def __init__(self, img, step=50, outfilename="SLICsuperpixel.img", stepsize=None, M=None, outparamfilename=None):
        if not M is None:
            self.M = M
            self.MM = M * M
        if stepsize is None:
            stepsize = (int(step), int(step))
        if outparamfilename is None:
            base, ext = os.path.splitext(outfilename)
            outparamfilename = base + "_params.dat"
        self.stepsize = stepsize
        self.filename = outfilename
        self.outparamfilename = outparamfilename
        self.img = img
        if self.DEBUGFLAG:
            cv2.imshow("test", self.img)
            cv2.waitKey(10)
        self.labimg = cv2.cvtColor(self.img, cv2.cv.CV_BGR2Lab)
        self.xylab = scipy.concatenate(
            ([[[x, y] for y in xrange(self.labimg.shape[1])] for x in xrange(self.labimg.shape[0])], self.labimg), 2)

    def _initialize_center_grid(self, cluster_size):
        xs = range(cluster_size[0] / 2, self.img.shape[0], cluster_size[0])
        ys = range(cluster_size[1] / 2, self.img.shape[1], cluster_size[1])
        return [scipy.array([x, y]) for x in xs for y in ys]

    def _getneighborhood(self, point2d, distanceset):
        '''
        Neighbor coordinates from points2d
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
        labs = self.labimg[points[0][0]:points[-1][0] + 1, points[0][1]:points[-1][1] + 1]

        return scipy.concatenate((points, labs.reshape((len(points), 3))), 1)

    def _search_minimum_gradient(self, point2d, distance):
        searchpoints = self._getneighborhood_in_image(point2d, [distance, distance])
        searchvals = [self.grad[point[0], point[1]] for point in searchpoints]
        return searchpoints[argmin(searchvals)][:2]

    def _initialize_center_avoidedge(self, centers, distance):
        self.grad = gradient_img(self.img)
        return [self._search_minimum_gradient(point, distance) for point in centers]

    def _initialize_center(self, cluster_size):
        centers = self._initialize_center_grid(cluster_size)
        centers = self._initialize_center_avoidedge(centers, self.INITCENTER_SEARCHDIFF)
        centers = [scipy.concatenate((center, self.labimg[center[0]][center[1]])) for center in centers]
        return centers

    def _initassignments(self):
        width, height = self.img.shape[:2]
        self.assignedindex = scipy.array([[0 for i in xrange(height)] for j in xrange(width)])
        self.assigneddistance = scipy.array([[self.INIFINITY_DISTANCE for i in xrange(height)] for j in xrange(width)])

    def calcdistance(self, point, center, spatialmax):
        '''
        Great Problem:
            Which distance should we use?
        '''
        ## -- new: L2norm optimized -- ##
        p1, p2, p3, p4, p5 = point
        c1, c2, c3, c4, c5 = center
        spatialdist = (c1 - p1) * (c1 - p1) + (c2 - p2) * (c2 - p2)
        colordist = (c3 - p3) * (c3 - p3) + (c4 - p4) * (c4 - p4) + (c5 - p5) * (c5 - p5)
        return colordist + spatialdist * self.MM / (spatialmax * spatialmax)

        ## -- old: euclid distance -- ##
        spatialdist = norm2d(center[:2] - point)
        spatialmax = max(stepsize)
        colordist = norm3d(center[2:] - self.labimg[point[0], point[1]])
        return colordist + spatialdist / spatialmax * self.M

    def calcdistance_mat(self, points, center, spatialmax):
        ## -- L2norm optimized -- ##
        center = scipy.array(center)
        difs = points - center
        norm = (difs ** 2).astype(float)
        norm[:, :, 0:2] *= (float(self.MM) / (spatialmax * spatialmax))
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
        sums = [scipy.zeros(5) for i in range(len(centers))]
        nums = [0 for i in range(len(centers))]
        width, height = self.img.shape[:2]
        print "E step"
        return [scipy.mean(self.xylab[self.assignedindex == i], 0) for i in xrange(len(centers))]

    def calcerror(self, centers, prevcenters):
        '''
        L2 norm of location
        '''
        print "error:", sum(
            [scipy.dot(now[:2] - prev[:2], now[:2] - prev[:2]) for now, prev in zip(centers, prevcenters)])
        return sum([scipy.dot(now[:2] - prev[:2], now[:2] - prev[:2]) for now, prev in zip(centers, prevcenters)])

    def iteration(self, centers, stepsize):
        error = sum([scipy.dot(center[:2], center[:2]) for center in centers])
        while error > self.ERROR_THRESHOLD:
            self.assignment(centers, stepsize)  ## M-step
            prevcenters, centers = centers, self.update(centers)  ## E-step
            error = self.calcerror(centers, prevcenters)
            print "L2 error:", error

            if self.DEBUGFLAG:
                base, ext = os.path.splitext(self.filename)
                self.filename = base.split("_error")[0] + "_error" + str(error) + ext
                self.resultimg(centers)
        return (centers, self.assignedindex)

    def resultimg(self, centers):
        print "show result"
        result = scipy.zeros(self.img.shape, scipy.uint8)
        width, height = result.shape[:2]
        colors = [scipy.array([int(random.uniform(0, 255)) for i in xrange(3)]) for j in xrange(len(centers))]
        for x in xrange(width):
            for y in xrange(height):
                result[x, y] = colors[self.assignedindex[x][y]]

        cv2.imshow("result", result)
        cv2.waitKey(10)
        cv2.imwrite("result.png", result)

    def saveparams(self, centers, filename=None):
        if filename is None: filename = self.outparamfilename
        import cPickle
        cPickle.dump((centers, self.assignedindex), open(filename, "w+"))

    def calc(self):
        centers = self._initialize_center(self.stepsize)
        self._initassignments()
        self.iteration(centers, self.stepsize)

        self.resultimg(centers)
        self.saveparams(centers)




if __name__ == '__main__':

    img_path='/home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'

    calculator = SlicCalculator(cv2.imread(img_path), step=50, M=10)

    # calculator.DEBUGFLAG = True
    calculator.calc()
