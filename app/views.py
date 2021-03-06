from django.shortcuts import render
from django.http import FileResponse
import sqlite3
import cv2 
import os
import math
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
import numpy as np;
# from matplotlib import pyplot as plt 
# from matplotlib import image as mpimg
from tkinter import * 

# -------------------------- change based on your path -------------------------- #
GREY_DIR = '/Users/tonycao/Desktop/csc664/csc664/app/static/app/images/grey/'
# ------------------------------------------------------------------------------- # 
# database images 
img_desc = []

class ShapeContext(object):

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        # maximum and minimum radius
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _hungarian(self, cost_matrix):
        """
            Here we are solving task of getting similar points from two paths
            based on their cost matrixes. 
            return total modification cost, indexes of matched points
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes

    def get_points_from_img(self, image, simpleto=100):
        """
            This is much faster version of getting shape points algo.
            It's based on cv2.findContours algorithm, which is basically returning shape points
            ordered by curve direction. So it's gives better and faster result
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        points = np.array(cnts[1][0]).reshape((-1, 2))
        if len(cnts[1]) > 1:
            points = np.concatenate([points, np.array(cnts[1][1]).reshape((-1, 2))], axis=0)
        points = points.tolist()
        step = len(points) / simpleto
        steps = int(step)
        lens = int(len(points))
        points = [points[i] for i in range(0, lens, steps)][:simpleto]
        if len(points) < simpleto:
            points = points + [[0, 0]] * (simpleto - len(points))
        return points


    def _cost(self, hi, hj):
        cost = 0
        for k in range(self.nbins_theta * self.nbins_r):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])

        return cost * 0.5

    def cost_by_paper(self, P, Q, qlength=None):
        p, _ = P.shape
        p2, _ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = np.zeros((p, p2))
        for i in range(p):
            for j in range(p2):
                C[i, j] = self._cost(Q[j] / d, P[i] / p)

        return C

    def compute(self, points):
        """
          Here we are computing shape context descriptor
        """
        t_points = len(points)
        # getting euclidian distance
        r_array = cdist(points, points)
        # getting two points with maximum distance to norm angle by them
        # this is needed for rotation invariant feature
        am = r_array.argmax()
        max_points = [am // t_points, am % t_points]
        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = np.zeros((t_points, t_points), dtype=int)
        # summing occurences in different log space intervals
        # logspace = [0.1250, 0.2500, 0.5000, 1.0000, 2.0000]
        # 0    1.3 -> 1 0 -> 2 0 -> 3 0 -> 4 0 -> 5 1
        # 0.43  0     0 1    0 2    1 3    2 4    3 5
        for m in range(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[max_points[0], max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((t_points, t_points)) - np.identity(t_points)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0

        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_q = (1 + np.floor(theta_array_2 / (2 * math.pi / self.nbins_theta))).astype(int)

        # building point descriptor based on angle and distance
        nbins = self.nbins_theta * self.nbins_r
        descriptor = np.zeros((t_points, nbins))
        for i in range(t_points):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in range(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            descriptor[i] = sn.reshape(nbins)

        return descriptor

    def cosine_diff(self, P, Q):
        """
            Fast cosine diff.
        """
        P = P.flatten()
        Q = Q.flatten()
        assert len(P) == len(Q), 'number of descriptors should be the same'
        return cosine(P, Q)

    def diff(self, P, Q, qlength=None):
        """
            More precise but not very speed efficient diff.
            if Q is generalized shape context then it compute shape match.
            if Q is r point representative shape contexts and qlength set to 
            the number of points in Q then it compute fast shape match.
        """
        result = None
        C = self.cost_by_paper(P, Q, qlength)

        result = self._hungarian(C)

        return result

  #---------------------------------------#      

def bin_img(image):
    img = cv2.imread(image)
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray, threshold1=100, threshold2=200)
    return edges


def list_files(dir): 
    # files
    fl_total = []
    # dir
    di = []

    for path, dirs, files in os.walk(dir):
        for dir in dirs:
            di.append(dir)

        fl = []
        for file in files:
            fl.append(file)
            if len(fl) == 5:
                fl_total.append(fl)

    return di, fl_total


def get_contour_bounding_rectangles(gray):
    """
      Getting all 2nd level bouding boxes based on contour detection algorithm.
    """
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in cnts[1]:
        (x, y, w, h) = cv2.boundingRect(cnt)
        res.append((x, y, x + w, y + h))

    return 


# Create your views here.
# request handler
def match_image(request):
    sc = ShapeContext() 
    # process query image
    post = request.POST.get("path")
    post = post.replace("/static/app/images/grey/", '')
    post = GREY_DIR + post

    # edge detection on query image
    img_query_edges = bin_img(post)

    # descriptor for query image
    # points = sc.get_points_from_img(img_query_edges, 15) # 15 X 15 matrix 
    # descriptor = sc.compute(points).flatten()
    # # print('hist: ', descriptor)
    # descriptor = np.array(descriptor)
    # # print('descriptor:', descriptor)
    img_query_hist = cv2.imread(post)
    hist_query = cv2.calcHist([img_query_hist], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_query[255, 255, 255] = 0
    cv2.normalize(hist_query, hist_query, 0, 1, cv2.NORM_MINMAX)

    descs = []
    points = sc.get_points_from_img(img_query_edges, 20)
    descriptor = sc.compute(points).flatten()
    # descs.append(descriptor)
    query_img_descriptor = np.array(descriptor)
    # print(type(descriptor_arr))

    # get DB image paths
    GREY_FILES = []

    dir, files = list_files(GREY_DIR)
    dir = [x for x in dir if not x.startswith('segmented')]
    dir = [GREY_DIR + s + "/org/" for s in dir]

    for a, b in zip(dir, files):
        for f in b:
            GREY_FILES.append(a + f)

    # compute SC for all images in DB 
    DB_DESCRIPTOR = {}
    
    for file in GREY_FILES:
        # edge detection 
        img = bin_img(file)

        # descriptor
        img_points = sc.get_points_from_img(img, 20) # 15 X 15 matrix 
        img_descriptor = sc.compute(img_points).flatten()
        img_descriptor_arr = np.array(img_descriptor)

        # key: image path, value: image descriptor
        if file not in DB_DESCRIPTOR:
            DB_DESCRIPTOR[file] = ''
        
        DB_DESCRIPTOR[file] = img_descriptor_arr

    scores = []
    for file in GREY_FILES:
        img_hist = cv2.imread(file)
        hist = cv2.calcHist([img_hist], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist[255, 255, 255] = 0
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        hist_diff = cv2.compareHist(hist_query, hist, cv2.HISTCMP_CORREL)

        scores.append((hist_diff, file))

    scores.sort(key=lambda y: y[0], reverse=True)
    
    best_match = {}
    for index, tuple in enumerate(scores):
        if tuple[1] not in best_match:
            best_match[tuple[1]] = ''
        best_match[tuple[1]] = tuple[0]

    # trim results to 10 
    best_match = dict(list(best_match.items())[:10])
    print('query image: ', post)
    
    for key in best_match:
        print('image: ' + str(key) + ' ---> ', best_match[key])
        
    # cost calculation
    COST_DICT = {}
    for path in DB_DESCRIPTOR: 
        compared_img_descriptor = DB_DESCRIPTOR[path]
        cost = sc._cost(query_img_descriptor, compared_img_descriptor)
        # print(cost)

        # key: image path, value: cost
        if path not in COST_DICT:
            COST_DICT[path] = ''
        
        COST_DICT[path] = cost
    
    COST_DICT = {key:val for key, val in COST_DICT.items() if val == 0}

    # print("query image: ", post)
    # print(COST_DICT)
    # best_match= {}

    # for cost in COST_DICT:
    #     min = sc._hungarian(cost)
    #     if image not in best_match:
    #         best_match[image] = ''
        
    #     best_match[image] = cost
   
    # print("query image: ", post)
    # for key, value in best_match.items() :
    #     print (value)

    #trim results 
    # best_match = dict(list(COST_DICT.items())[:5])
    # print("query image: ", post)
    # print('best match: ', best_match)

    return render(request, 'hello.html', {'context': best_match})


def load_front_page(request):
    try:
        GREY_FILES = []

        dir, files = list_files(GREY_DIR)
        dir = [x for x in dir if not x.startswith('segmented')]
        dir = [GREY_DIR + s + "/org/" for s in dir]

        for a, b in zip(dir, files):
            for f in b:
                GREY_FILES.append(a + f)

        #print(GREY_FILES)

        image_paths = {}
        for file in GREY_FILES:
            if file not in image_paths:
                image_paths[file] = ''
            image_paths[file] = file.replace('/Users/tonycao/Desktop/csc664/csc664/app', '')

        # print(image_paths)
        
        return render(request, 'hello.html', {'context': image_paths})
    except:
        print("Can not open file system.")