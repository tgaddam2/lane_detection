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
        self.pub = rospy.Publisher('frame', Image,queue_size=10)
        self.got_first = False
        self.RESTARTED = False
    def load_cam(self):
        # self.vidObj = cv2.VideoCapture('/home/tgaddam/catkin_ws/src/lane_detection/nodes/harder_challenge_video.mp4')
        self.vidObj = cv2.VideoCapture('/home/tgaddam/catkin_ws/src/lane_detection/nodes/test_videos/Lane Detection Test Video 01.mp4')
        
    def start(self):
        while not rospy.is_shutdown():
            ret, image = self.vidObj.read()
            
            # image = self.br.imgmsg_to_cv2(image)
                                    
            if ret:
                
                if not self.got_first:
                    rospy.loginfo("GOT FIRST IMAGE" if not self.RESTARTED else "RESTARTING VIDEO")
                self.got_first = True
                # rospy.loginfo(type(image))
                
                self.pub.publish(self.br.cv2_to_imgmsg(image))
                
            else:
                self.load_cam()
                self.RESTARTED = True
                self.got_first = False
            
if __name__ == '__main__':
    rospy.init_node("image_node", anonymous=True)
    my_node = Node()
    my_node.start()