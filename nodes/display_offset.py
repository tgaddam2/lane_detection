#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

import lane_util as lane_util
import simple_lane_util as simple_lane_util

offset_topic = '/offset_visual'

offset_value_topic = '/offset'
cam_topic = '/camera/rgb/image_rect_color' # /frame for mp4

class Node(object):
    def __init__(self):
        # self.cam = cv2.VideoCapture(0)
        self.br = CvBridge()
        self.vid_sub = rospy.Subscriber(cam_topic, Image, callback=self.img_callback, queue_size=10)
        self.offset_sub = rospy.Subscriber(offset_value_topic, Float64, callback=self.offset_callback, queue_size=10)
        self.offset = 0
        
        self.vid_edited_pub = rospy.Publisher(offset_topic, Image,queue_size=10)

    def img_callback(self, image):
        self.image = self.br.imgmsg_to_cv2(image)
        
        self.processImg()
        
        self.vid_edited_pub.publish(self.br.cv2_to_imgmsg(self.image))
        
    def offset_callback(self, offset):
        self.offset = offset.data
    
    def processImg(self):
        # rospy.loginfo(f"y: {self.image.shape[0]}        x: {self.image.shape[1]}")
        
        
        center_x = int(self.image.shape[1]/2)
        center_y = int(self.image.shape[0]/2)
        
        offset_x = center_x + int(center_x * self.offset)
        
        self.image = cv2.circle(self.image, (center_x, center_y), 5, (0,0,255), -1)
        
        self.image = cv2.line(self.image, (center_x, center_y), (offset_x, center_y), (0,0,255), 3)
        
        self.image = simple_lane_util.offset_return(self.image)[0]
         
            
if __name__ == '__main__':
    rospy.init_node("display_offset", anonymous=True)
    my_node = Node()
    rospy.loginfo("initialized")
    rospy.spin()