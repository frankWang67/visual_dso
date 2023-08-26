import cv2 as cv
import numpy as np
import queue
from dso.vision_module.object_tracking import find_contours, track_objects, get_angle

# DATA_PER_OBJECT = 3
DATA_PER_OBJECT = 5

class VisionObserver:
    def __init__(self, colors=None, stored_frames_num=3):
        self.stored_frames_num = stored_frames_num
        self.data_queue = queue.Queue(maxsize=stored_frames_num)
        if colors is None:
            self.colors = np.random.randint(0, 255, size=(10, 3), dtype=np.uint8)
        else:
            self.colors = colors
    
    def start(self, first_frame):
        self.trackers_setup(first_frame)
        return self.data
        
    def trackers_setup(self, first_frame):
        self.trackers = []
        rects, self.res = find_contours(first_frame, self.colors)
        self.object_num = len(rects)
        self.data = np.zeros(self.stored_frames_num * DATA_PER_OBJECT * self.object_num)
        frame_data = []
        for rect in rects:
            tracker = cv.legacy.TrackerCSRT_create()
            tracker.init(image=first_frame, boundingBox=rect)
            self.trackers.append(tracker)
            angle, box = get_angle(first_frame, rect)
            if angle is not None:
                cv.drawContours(self.res, [box], 0, (0, 255, 0), 3)
            else:
                angle = 0.0
            object_data = np.append(rect, angle)
            # object_data = np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2], dtype=np.int)
            # object_data = np.append(object_data, angle)

            frame_data = np.concatenate((frame_data, object_data))
        while not self.data_queue.full():
            self.data_queue.put(frame_data)
        for i in range(self.stored_frames_num):
            start_idx = i * DATA_PER_OBJECT * self.object_num
            end_idx = (i + 1) * DATA_PER_OBJECT * self.object_num
            self.data[start_idx:end_idx] = frame_data

    def step(self, next_frame):
        rects, self.res = track_objects(next_frame, self.trackers, self.colors)
        frame_data = []
        for rect in rects:
            angle, box = get_angle(next_frame, rect)
            if angle is not None:
                cv.drawContours(self.res, [box], 0, (0, 255, 0), 3)
            else:
                angle = 0.0
            object_data = np.append(rect, angle)
            # object_data = np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2], dtype=np.int)
            # object_data = np.append(object_data, angle)

            frame_data = np.concatenate((frame_data, object_data))
        self.data_queue.get()
        self.data_queue.put(frame_data)
        for i in range(self.stored_frames_num - 1):
            start_idx_old = i * DATA_PER_OBJECT * self.object_num
            end_idx_old = (i + 1) * DATA_PER_OBJECT * self.object_num
            start_idx_new = (i + 1) * DATA_PER_OBJECT * self.object_num
            end_idx_new = (i + 2) * DATA_PER_OBJECT * self.object_num
            self.data[start_idx_old:end_idx_old] = self.data[start_idx_new:end_idx_new]
        start_idx = (self.stored_frames_num - 1) * DATA_PER_OBJECT * self.object_num
        end_idx = self.stored_frames_num * DATA_PER_OBJECT * self.object_num
        self.data[start_idx:end_idx] = frame_data
        return self.data

    def render(self):
        frame = cv.cvtColor(self.res, cv.COLOR_RGB2BGR)
        cv.imshow("env", frame)
        if cv.waitKey(10) == 27:
            return False
        return True