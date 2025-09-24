#!/usr/bin/env python2
import time
import rospy
from sensor_msgs.msg import Image
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
cvBridge = CvBridge()

from rospy.numpy_msg import numpy_msg
from main.msg import Int16, Float32


class ConvertImage:
    def __init__(self):
        self.cam1_image = None
        self.cam2_image = None
        self.cam3_image = None
        self.cam1_image_msg = None
        self.cam2_image_msg = None
        self.cam3_image_msg = None
        self.cam1_depth = None
        self.cam2_depth = None
        self.cam1_depth_msg = None
        self.cam2_depth_msg = None
        self.fr = 0
        self.recording = 2

        rospy.init_node('imgmsg_converter', anonymous=True)

        rospy.Subscriber("/camera1/color/image_raw", Image, self.callback1)
        rospy.Subscriber("/camera2/color/image_raw", Image, self.callback2)
        
        rospy.Subscriber("/camera1/aligned_depth_to_color/image_raw", Image, self.callback1d)
        rospy.Subscriber("/camera2/aligned_depth_to_color/image_raw", Image, self.callback2d)
        print("Subscribed to camera topics")
        

        time.sleep(2.0) # wait for rgb image messages
        print("Initialized")

    def callback1(self, data):
        self.cam1_image = cvBridge.imgmsg_to_cv2(data, 'bgr8')
        self.cam1_image_msg = self.cam1_image.astype(np.int16).reshape(-1)

    def callback2(self, data):
        self.cam2_image = cvBridge.imgmsg_to_cv2(data, 'bgr8')
        self.cam2_image_msg = self.cam2_image.astype(np.int16).reshape(-1)

    def callback3(self, data):
        self.cam3_image = cvBridge.imgmsg_to_cv2(data, 'bgr8')
        # flip x and y
        self.cam3_image = cv2.flip(self.cam3_image, -1)
        self.cam3_image_msg = self.cam3_image.astype(np.int16).reshape(-1)
    
    def callback1d(self, data):
        self.cam1_depth = cvBridge.imgmsg_to_cv2(data, 'passthrough')
        self.cam1_depth_msg = self.cam1_depth.astype(np.float32).reshape(-1)
    
    def callback2d(self, data):
        self.cam2_depth = cvBridge.imgmsg_to_cv2(data, 'passthrough')
        self.cam2_depth_msg = self.cam2_depth.astype(np.float32).reshape(-1)

    def imgmsg_converter(self):

        rate = rospy.Rate(30)

        pub1 = rospy.Publisher('/camera1/image_list', numpy_msg(Int16), queue_size=10)
        pub2 = rospy.Publisher('/camera2/image_list', numpy_msg(Int16), queue_size=10)
        pub1d = rospy.Publisher('/camera1/depth_list', numpy_msg(Float32), queue_size=10)
        pub2d = rospy.Publisher('/camera2/depth_list', numpy_msg(Float32), queue_size=10)

        while not rospy.is_shutdown():

            pub1.publish(self.cam1_image_msg)
            pub2.publish(self.cam2_image_msg)
            pub1d.publish(self.cam1_depth_msg)
            pub2d.publish(self.cam2_depth_msg)

            rate.sleep()

        rospy.spin()

if __name__ == '__main__':
    ci = ConvertImage()
    ci.imgmsg_converter()