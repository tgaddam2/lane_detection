#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

import lane_util as lane_util

offset_topic = '/offset_visual'
cam_topic = '/camera/rgb/image_rect_color' # /frame for mp4

class Node(object):
    def __init__(self):
        # self.cam = cv2.VideoCapture(0)
        self.br = CvBridge()
        self.vid_sub = rospy.Subscriber(cam_topic, Image, callback=self.callback, queue_size=10)
        self.vid_edited_pub = rospy.Publisher(offset_topic, Image, queue_size=10)

    def callback(self, image):
        self.image = self.br.imgmsg_to_cv2(image)
        
        offset, self.image = self.processImg()
        
        if offset is None:
            return
        
        height, width = self.image.shape[:2]
        
        center_x = width / 2
        center_y = height / 2
        offset_x = center_x + int(center_x * offset)
        
        self.image = cv2.circle(self.image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
        self.image = cv2.line(self.image, (int(center_x), int(center_y)), (int(offset_x), int(center_y)), (255, 0, 0), 3)
        
        self.vid_edited_pub.publish(self.br.cv2_to_imgmsg(self.image))
    
    def processImg(self):
        # rospy.loginfo(f"y: {self.image.shape[0]}        x: {self.image.shape[1]}")
        offset = lane_util.process_and_draw(self.image, visualize=True)[0]
        final_image = lane_util.process_and_draw(self.image, visualize=True)[1]
        return offset, final_image
         
            
if __name__ == '__main__':
    rospy.init_node("display_offset", anonymous=True)
    my_node = Node()
    rospy.loginfo("initialized")
    rospy.spin()