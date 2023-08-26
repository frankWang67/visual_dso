import cv2 as cv
import numpy as np

def find_contours(img, colors):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img_blurred = cv.GaussianBlur(img_gray, (13, 13), 0)
    img_canny = cv.Canny(img_blurred, 50, 150)
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    res = img.copy()
    rects = []
    for i in range(len(contours)):
        cnt = contours[i]
        if cv.contourArea(cnt) < 1:
            continue
        # convert data types int64 to int
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        rect = cv.boundingRect(cnt)
        rect = list(rect)
        rect[0] -= 10
        rect[1] -= 10
        rect[2] += 20
        rect[3] += 20
        rect = tuple(map(int, rect))
        rects.append(rect)
        cv.rectangle(res, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, 3)

    return rects, res

def track_objects(img, trackers, colors):
    res = img.copy()
    rois = []
    
    for i in range(len(trackers)):
        tracker = trackers[i]
        # convert data types int64 to int
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        success, roi = tracker.update(img)
        roi = tuple(map(int, roi))
        if success:
            rois.append(roi)
            cv.rectangle(res, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), color, 3)
        else:
            print("Tracking failed!")

    return rois, res

def get_angle(img, region):
    region = list(region)
    temp1 = np.clip(region[1], 0, img.shape[1])
    region[3] -= temp1 - region[1]
    region[1] = temp1
    temp0 = np.clip(region[0], 0, img.shape[0])
    region[2] -= temp0 - region[0]
    region[0] = temp0
    sub_img = img[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
    gray = cv.cvtColor(sub_img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (11, 11), 0)
    canny = cv.Canny(blurred, 50, 150)
    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found in sub-image!")
        return None, None
    cnt_idx = np.argmax([cv.arcLength(c, False) for c in contours])
    rect = cv.minAreaRect(contours[cnt_idx])
    angle = rect[2]
    box = cv.boxPoints(rect)
    box = np.int0(box)
    box[:, 0] += region[0]
    box[:, 1] += region[1]
    return angle, box