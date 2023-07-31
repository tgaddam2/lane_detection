#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

import lane_util as lane_util

offset_topic = '/offset'
cam_topic = '/camera/rgb/image_rect_color' # # /frame for mp4

class Node(object):
    def __init__(self):
        # self.cam = cv2.VideoCapture(0)
        self.br = CvBridge()
        self.vid_sub = rospy.Subscriber(cam_topic, Image, callback=self.callback, queue_size=10)
        self.offset_pub = rospy.Publisher(offset_topic, Float64,queue_size=10)
    def callback(self, image):
        self.image = self.br.imgmsg_to_cv2(image)
        # rospy.loginfo(self.image.shape)
        offset = self.processImg()
        # rospy.loginfo(offset)

        if offset is None:
            return
        # rospy.loginfo(offset)
        msg = Float64()
        msg.data = offset
        self.offset_pub.publish(msg)
        
    
    def processImg(self):
        offset = lane_util.process_and_draw(self.image)
        return offset
                
            
if __name__ == '__main__':
    rospy.init_node("lane_offset", anonymous=True)
    my_node = Node()
    rospy.loginfo("initialized")
    rospy.spin()