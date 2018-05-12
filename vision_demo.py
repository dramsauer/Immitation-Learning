import numpy as np
import cv2
import copy

def threshold(img, threshold):
    "Thresholds an image"
    _i = 0
    _j = 0
    while _i < img.shape[0]:
        _j = 0
        while _j < img.shape[1]:
            if (img.item(_i, _j, 0) + img.item(_i, _j, 1) + img.item(_i, _j, 2)) > threshold:
                img[_i,_j][0] = 255
                img[_i,_j][1] = 255
                img[_i,_j][2] = 255
            else:
                img[_i,_j][0] = 0
                img[_i,_j][1] = 0
                img[_i,_j][2] = 0
            _j += 1
        _i += 1
    print("Threshold done.")
    return


def extractBlueRGB(img):
    "Extracts blue channel"

    _i = 0
    _j = 0
    while _i < img.shape[0]:
        _j = 0
        while _j < img.shape[1]:
            #img[_i,_j][0] = 0
            img[_i,_j][1] = 0
            img[_i,_j][2] = 0
            _j += 1
        _i += 1

    print("Channel extraction via RGB done.")

    return img

def extractBlueHSV(img):
    "Extracts blue channel based on HSV"

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    img = cv2.bitwise_and(img, img, mask=mask)

    print("Channel extraction via HSV done.")

    return img


def thresholdOtsu(img):
    "Thresholds an image based on blue channel"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if img[0,0] == 0:
        cv2.bitwise_not(img, img);

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print("Otsu\'s Threshold done: " + str(th))

    return img


def searchtemplate(img, template):
    "Searches for template in image"

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    c, w, h = template.shape[::-1]

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, (0,0,255), 10)

    print("Template search done.")

    return img


def regiongrowing(img):
    "Region grows an image"
    _i = 1
    _j = 1
    img1 = copy.deepcopy(img)
    while _i < img.shape[0]-1:
        _j = 0
        while _j < img.shape[1]-1:
            if (img.item(_i, _j, 0) == 0 and img.item(_i, _j, 1) == 0 and img.item(_i, _j, 2) == 0):
                img1[_i-1, _j-1][0] = 0
                img1[_i-1, _j-1][1] = 0
                img1[_i-1, _j-1][2] = 0
                img1[_i  , _j-1][0] = 0
                img1[_i  , _j-1][1] = 0
                img1[_i  , _j-1][2] = 0
                img1[_i+1, _j-1][0] = 0
                img1[_i+1, _j-1][1] = 0
                img1[_i+1, _j-1][2] = 0

                img1[_i-1, _j][0] = 0
                img1[_i-1, _j][1] = 0
                img1[_i-1, _j][2] = 0
                img1[_i  , _j][0] = 0
                img1[_i  , _j][1] = 0
                img1[_i  , _j][2] = 0
                img1[_i+1, _j][0] = 0
                img1[_i+1, _j][1] = 0
                img1[_i+1, _j][2] = 0

                img1[_i-1, _j+1][0] = 0
                img1[_i-1, _j+1][1] = 0
                img1[_i-1, _j+1][2] = 0
                img1[_i  , _j+1][0] = 0
                img1[_i  , _j+1][1] = 0
                img1[_i  , _j+1][2] = 0
                img1[_i+1, _j+1][0] = 0
                img1[_i+1, _j+1][1] = 0
                img1[_i+1, _j+1][2] = 0
            _j += 1
        _i += 1

    print("Region growing done.")
    return img1


def track(img):
    "Counts cluster"
    img1 = copy.deepcopy(img)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    connectivity = 24
    cv2.bitwise_not(thresh, thresh)
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    num_labels = output[0]
    stats = output[2]

    noCluster = []
    yesCluster = []
    for _c in range(num_labels):
        if (stats[_c, cv2.CC_STAT_AREA] < 10000) or (stats[_c, cv2.CC_STAT_AREA] > 300000):
            noCluster.append(_c)
        else:
            yesCluster.append(_c)

    _i = 0
    _j = 0
    while _i < img.shape[0]:
        _j = 0
        while _j < img.shape[1]:
            if (output[1][_i][_j] == 0):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 255
                img[_i, _j][2] = 255
            elif (output[1][_i][_j] in noCluster):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 255
                img[_i, _j][2] = 255
            elif (output[1][_i][_j] in yesCluster):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 0
                img[_i, _j][2] = 0
            else:
                img[_i, _j][0] = 255/num_labels*output[1][_i][_j]
                img[_i, _j][1] = 255/num_labels*output[1][_i][_j]
                img[_i, _j][2] = 255/num_labels*output[1][_i][_j]
            _j += 1
        _i += 1

    x1 = stats[yesCluster[0], cv2.CC_STAT_LEFT]
    x2 = stats[yesCluster[0], cv2.CC_STAT_LEFT] +  stats[yesCluster[0], cv2.CC_STAT_WIDTH]
    y1 = stats[yesCluster[0], cv2.CC_STAT_TOP]
    y2 = stats[yesCluster[0], cv2.CC_STAT_TOP] + stats[yesCluster[0], cv2.CC_STAT_HEIGHT]

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 10)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Position: (" + str(x1+(x2-x1)/2) + "," + str(y1+(y2-y1)/2) + ")", (10,730), font, 2, (0,0,0), 2, cv2.LINE_AA)

    print("Clustering done. Found " + str(len(yesCluster)) + " cluster.")
    return img


def cluster(img):
    "Counts cluster"
    img1 = copy.deepcopy(img)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    connectivity = 24
    cv2.bitwise_not(thresh, thresh)
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    num_labels = output[0]
    stats = output[2]

    noCluster = []
    yesCluster = []
    for _c in range(num_labels):
        if (stats[_c, cv2.CC_STAT_AREA] < 100) or (stats[_c, cv2.CC_STAT_AREA] > 300000):
            noCluster.append(_c)
        else:
            yesCluster.append(_c)

    _i = 0
    _j = 0
    while _i < img.shape[0]:
        _j = 0
        while _j < img.shape[1]:
            if (output[1][_i][_j] == 0):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 255
                img[_i, _j][2] = 255
            elif (output[1][_i][_j] in noCluster):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 255
                img[_i, _j][2] = 255
            elif (output[1][_i][_j] == 1):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 0
                img[_i, _j][2] = 0
            elif (output[1][_i][_j] == 2):
                img[_i, _j][0] = 0
                img[_i, _j][1] = 255
                img[_i, _j][2] = 0
            elif (output[1][_i][_j] == 3):
                img[_i, _j][0] = 0
                img[_i, _j][1] = 0
                img[_i, _j][2] = 255
            elif (output[1][_i][_j] == 4):
                img[_i, _j][0] = 0
                img[_i, _j][1] = 255
                img[_i, _j][2] = 255
            elif (output[1][_i][_j] == 5):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 0
                img[_i, _j][2] = 255
            elif (output[1][_i][_j] == 6):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 255
                img[_i, _j][2] = 0
            elif (output[1][_i][_j] == 7):
                img[_i, _j][0] = 255
                img[_i, _j][1] = 128
                img[_i, _j][2] = 128
            elif (output[1][_i][_j] == 8):
                img[_i, _j][0] = 128
                img[_i, _j][1] = 128
                img[_i, _j][2] = 128
            else:
                img[_i, _j][0] = 255/num_labels*output[1][_i][_j]
                img[_i, _j][1] = 255/num_labels*output[1][_i][_j]
                img[_i, _j][2] = 255/num_labels*output[1][_i][_j]
            _j += 1
        _i += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Found " + str(len(yesCluster)) + " objects.",(10,730), font, 2, (0,0,0), 2, cv2.LINE_AA)

    print("Clustering done. Found " + str(len(yesCluster)) + " cluster.")
    return img

img = cv2.imread('./demo_image.jpg',3)
template = cv2.imread('./template.jpg',3)
cv2.imshow('image',img)
k = cv2.waitKey(0)

while (k != 27):
    print("\n 0 load original\n\
 1 Threshold r+b+g > 440\n\
 2 Region growing\n\
 3 Connected Component Analysis and count\n\
 4 Extract b channel\n\
 5 Extract blue in HSV\n\
 6 Convert to gray and Otsu\'s Threshold\n\
 7 Connected Component Analysis and position of largest cluster\n\
 8 Template matching")
    if k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('save.png',img)
        cv2.destroyAllWindows()
    elif k == ord('0'): # original
        print("Loaded original file from disk.")
        img = cv2.imread('./demo_image.jpg', 3)
        cv2.imshow('image',img)
    elif k == ord('1'): # threshold
        threshold(img, 440)
        cv2.imshow('image',img)
    elif k == ord('2'): # region growing
        img = regiongrowing(img)
        cv2.imshow('image',img)
    elif k == ord('3'): # cluster
        img = cluster(img)
        cv2.imshow('image',img)
    elif k == ord('4'): # extract blue channel via RGB
        img = extractBlueRGB(img)
        cv2.imshow('image',img)
    elif k == ord('5'): # extract blue channel via HSV
        img = extractBlueHSV(img)
        cv2.imshow('image',img)
    elif k == ord('6'): # Otsu's Threshold
        img = thresholdOtsu(img)
        cv2.imshow('image',img)
    elif k == ord('7'): # Track largest cluster
        img = track(img)
        cv2.imshow('image',img)
    elif k == ord('8'): # Search template
        img = searchtemplate(img, template)
        cv2.imshow('image',img)

    k = cv2.waitKey(0)

cv2.destroyAllWindows()
