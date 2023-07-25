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
        self.pub = rospy.Publisher('webcam', Image,queue_size=10)


    def start(self):
        while not rospy.is_shutdown():
            image = self.cam.read()[1]
            self.pub.publish(self.br.cv2_to_imgmsg(image))
            
if __name__ == '__main__':
    rospy.init_node("image_node", anonymous=True)
    my_node = Node()
    my_node.start()