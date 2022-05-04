from django.shortcuts import render
from django.http import FileResponse
import sqlite3
import cv2 
import os
import numpy as np;
import matplotlib.pyplot as plt

# -------------------------- change based on your path -------------------------- #
GREY_DIR = '/Users/tonycao/Desktop/csc664/csc664/app/static/app/images/grey/'
# ------------------------------------------------------------------------------- # 

GREY_FILES = []

def bin_img(image):
    img = cv2.imread(image)
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
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


# Create your views here.
# request handler
def match_image(request):
    try:

        # process query image
        post = request.POST.get("path")
        post = post.replace("/static/app/images/grey/", '')
        post = GREY_DIR + post

        # edge detection on query image
        img_query_edges = bin_img(post)

        # compute shape context 
        # for each point on the edge, make coarse histogram in log-polar (shape context)
    
    except: 
        print("error while matching images.")


def load_front_page(request):
    try:
        dir, files = list_files(GREY_DIR)
        dir = [x for x in dir if not x.startswith('segmented')]
        dir = [GREY_DIR + s + "/org/" for s in dir]

        for a, b in zip(dir, files):
            for f in b:
                GREY_FILES.append(a + f)

        # print(GREY_FILES)

        image_paths = {}
        for file in GREY_FILES:
            if file not in image_paths:
                image_paths[file] = ''
            image_paths[file] = file.replace('/Users/tonycao/Desktop/csc664/csc664/app', '')

        # print(image_paths)
        
        return render(request, 'hello.html', {'context': image_paths})
    except:
        print("Can not open file system.")