#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

import 1_lane_line

offset_topic = '/offset'
vid_topic = '/mp4'

class Node(object):
    def __init__(self):
        # self.cam = cv2.VideoCapture(0)
        self.br = CvBridge()
        self.vid_sub = rospy.Subscriber(vid_topic, Image, callback=self.callback, queue_size=10)
        self.offset_pub = rospy.Publisher(offset_topic, Float64,queue_size=10)
    def callback(self, image):
        self.image = self.br.imgmsg_to_cv2(image)
        offset = self.processImg()
        if offset is None:
            return
        # rospy.loginfo(offset)
        msg = Float64()
        msg.data = offset
        self.offset_pub.publish(msg)
        
    
    def processImg(self):
        self.left_lane, self.right_lane = lane_line.lane_cords(self.image)
        if self.left_lane is None or self.right_lane is None:
            return None
        rows, cols = self.image.shape[:2]
        
        line_pnts = np.asarray([self.left_lane, self.right_lane])
        
        camera_pos = cols * 0.5
        # lane_center = (self.left_lane[0][0] + self.right_lane[0][0]) / 2
        lane_center = np.average(line_pnts[:,0])
        
        return abs(lane_center - camera_pos)
        
        
            
if __name__ == '__main__':
    rospy.init_node("image_node", anonymous=True)
    my_node = Node()
    rospy.spin()