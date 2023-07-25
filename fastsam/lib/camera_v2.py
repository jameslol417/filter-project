#!/usr/bin/env python
import rospy
import roslib
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


from geometry_msgs.msg import PointStamped
import tf


class Camera:

    def __init__(self, callback = None, click = None):
        print('Camera')
        self.bridge_ros2cv = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.__imageCallback, queue_size = 1000)
        self.__callback = callback
        self.__click = click
        self.__image = None
        self.__listener = None
        self.__click_x = 0
        self.__click_y = 0
        self.width = 640
        self.height = 640



    def start(self):
        print("starting camera")
        #rospy.init_node('camera_show', anonymous=True)
        # msg=rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        # print(msg)

        # self.fx = msg.K[0]
        # self.fy = msg.K[4]
        # self.cx = msg.K[2]
        # self.cy = msg.K[5]

        self.fx = 570.3422241210938*2
        self.fy = 570.3422241210938*2
        self.cx = 319.5
        self.cy = 239.5


        self.__listener = tf.TransformListener()
        self.__listener.waitForTransform("/base_link", "/camera_color_optical_frame", rospy.Time(0),rospy.Duration(4.0))


    def convert2d_3d(self, u,v):
        x =  ( u  - self.cx )/ self.fx
        y =  ( v  - self.cy )/ self.fy
        z = 1.0
        return (x,y,z)

    def convert3d_2d(self,x,y,z):
        # 3d point on camera -> 2d point on camera
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

    def convert3d_3d(self,x,y,z):
        # point from camera -> point from base
        cam_point = PointStamped()
        cam_point.header.frame_id = "/camera_color_optical_frame"
        cam_point.header.stamp = rospy.Time(0)
        cam_point.point.x = x
        cam_point.point.y = y
        cam_point.point.z = z
        self.__listener.waitForTransform("/base_link", "/camera_color_optical_frame", rospy.Time(0),rospy.Duration(4.0))
        p = self.__listener.transformPoint("/base_link", cam_point)
        return (p.point.x,p.point.y,p.point.z)

    def showImage(self, frame, frame_name = 'Frame'):
        cv2.imshow(frame_name, frame[...,::-1])
        cv2.setMouseCallback(frame_name, self.onMouse)
        cv2.waitKey(1)

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.__click is not None:
                self.__click(self,x,y)
            self.__click_x = x
            self.__click_y = y

    def __imageCallback(self, image_msg):
        frame = self.bridge_ros2cv.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        self.__image = frame
        if self.__callback == None:
            _frame = frame
        else:
            _frame = self.__callback(frame)
            frame = _frame

    def getClickPoint(self):
        return(self.__click_x,self.__click_y)
    def getImage(self):
        if type(self.__image) == type(None):
            return np.zeros((480, 640, 3), np.uint8)
        return self.__image.copy()
