# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict


detection_graph = tf.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def find_finger(img):
    crop_image = img
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([0,30,80]), np.array([20,255,255]))

    # Kernel for morphological transformation
    kernel = np.ones((5,5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    #image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    # Get five largest contour area
    cnt = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    for contour in cnt:
        # Create bounding rectangle around the contour
        x,y,w,h = cv2.boundingRect(contour)
        if w>100 and h>100:
            cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)

            # Find convex hull
            hull = cv2.convexHull(contour)

            # Draw contour
            drawing = np.zeros(crop_image.shape,np.uint8)
            cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
            cv2.drawContours(drawing,[hull],-1,(0,0,255),0)

            # Find convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour,hull)

            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
            # tips) for all defects
            count_defects = 0

            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

                # if angle > 90 draw a circle at the far point
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_image,far,1,[0,0,255],-1)

                cv2.line(crop_image,start,end,[0,255,0],2)

            # Print number of fingers
            if count_defects == 0:
                cv2.putText(img,"1", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX , 1, 2)
            elif count_defects == 1:
                cv2.putText(img,"2", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            elif count_defects == 2:
                cv2.putText(img, "3", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            elif count_defects == 3:
                cv2.putText(img,"4", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            elif count_defects == 4:
                cv2.putText(img,"5", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            else:
                print (count_defects)

    # Show required images
    cv2.imshow("Result", img)
    #cv2.waitKey(500)
    return crop_image

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):

        if (scores[i] > score_thresh):

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

            part_image = image_np[int(top):int(bottom), int(left):int(right)]
            out_part_image = find_finger(part_image)

            cv2.imshow("temporal" + str(i), out_part_image)

            print((left, right, top, bottom))
            #cv2.waitKey(2000)

    #cv2.destroyAllWindows()
def make_real_bbox(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    bbox = []

    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

            part_image = image_np[int(top):int(bottom), int(left):int(right)]
            out_part_image = find_finger(part_image)
            bbox.append((left, right, top, bottom))

    return bbox


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
