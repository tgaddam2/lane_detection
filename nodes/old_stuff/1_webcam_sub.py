#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

class Node(object):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.br = CvBridge()
        self.pub = rospy.Subscriber('webcam', Image,callback=self.callback,queue_size=10)

    def callback(self, image):
        self.image = self.br.imgmsg_to_cv2(image)
    
    def processImg(self):
        pass
            
if __name__ == '__main__':
    rospy.init_node("image_node", anonymous=True)
    my_node = Node()
    rospy.spin()