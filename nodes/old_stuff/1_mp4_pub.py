#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np



class Node():
    def __init__(self):
        self.load_cam()
        self.br = CvBridge()
        self.pub = rospy.Publisher('mp4', Image,queue_size=10)
        self.got_first = False
    def load_cam(self):
        self.vidObj = cv2.VideoCapture('/home/tgaddam/catkin_ws/src/lane_detection/nodes/test2.mp4')
    def start(self):
        while not rospy.is_shutdown():
            ret, image = self.vidObj.read()
            if ret:
                if not self.got_first:
                    rospy.loginfo("GOT FIRST IMAGE")
                self.got_first = True
                # rospy.loginfo(type(image))
                
                image = cv2.circle(image, (int(image.shape[1]/2), int(image.shape[0]/2)), 10, (0,0,255), -1)
                self.pub.publish(self.br.cv2_to_imgmsg(image))
            else:
                self.load_cam()
                self.got_first = False
            
if __name__ == '__main__':
    rospy.init_node("image_node", anonymous=True)
    my_node = Node()
    my_node.start()