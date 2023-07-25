#!/usr/bin/env python


import rospy
import roslib
from std_msgs.msg import Float32MultiArray, String, MultiArrayLayout, MultiArrayDimension, Int32
from control_msgs.msg import PointHeadAction, PointHeadGoal, PointHeadActionResult
from actionlib_msgs.msg import GoalStatusArray
import actionlib

deg2rad = 0.0174533

class Robot:

    def __init__(self):
        print('Initializing robot')
        self.__cmdPub = rospy.Publisher("/gretchen/joint/cmd", Float32MultiArray, queue_size = 10)
        self.__lookatpointPub = rospy.Publisher("/look_at_point", Float32MultiArray, queue_size = 10)

        rospy.Subscriber("/gretchen/joint/poses", Float32MultiArray, self.__jointCallback, queue_size = 20)
        rospy.Subscriber("/head_controller/absolute_point_head_action/result", PointHeadActionResult, self.__actionResultCallback, queue_size = 10)
        self.cur_pan_angle = 0
        self.cur_tilt_angle = 0


        # motion result
        self.__isMotion = False

        # action status
        self.__motor_data_status = -1
        self.__action_status = -1
        self.__allow_action = 1
        rospy.Subscriber("/head_controller/absolute_point_head_action/status", GoalStatusArray, self.__actionStatusCallback, queue_size = 10 )
        rospy.Subscriber("/motor_data_status", Int32, self.__motorDataStatusCallback, queue_size = 10 )


    def start(self):
        print ("Starting robot")

        self.__cmd_pan = self.getPanAngle()
        self.__cmd_tilt = self.getTiltAngle()
        self.__max_joint_speed = rospy.get_param('~max_joint_speed', 0.1)
        self.__max_pan_angle_radian = rospy.get_param("~max_pan_angle_radian", 1.0)
        self.__max_tilt_angle_radian = rospy.get_param("~max_tilt_angle_radian", 1.0)
        self.__initParam()

    def __actionStatusCallback(self, action_status):
        if len(action_status.status_list)>1:
            #print(action_status.status_list[1].status)
            self.__action_status = action_status.status_list[1].status
            if(self.__action_status == 1):
                self.__allow_action = -1

    def __motorDataStatusCallback(self, motor_data_status_msg):
        self.__motor_data_status = motor_data_status_msg.data
        if(self.__motor_data_status == 1):
            self.__allow_action = 1
        #print(self.__motor_data_status)

    def getPosition(self):
        return [self.cur_pan_angle, self.cur_tilt_angle]



    def lookatpoint(self, x, y, z, velocity=10.8):
        if self.__allow_action == 1:
            head_client = actionlib.SimpleActionClient("/head_controller/absolute_point_head_action", PointHeadAction)
            head_client.wait_for_server()
            goal = PointHeadGoal()
            goal.target.header.stamp = rospy.Time.now()
            goal.target.header.frame_id = "/camera_color_optical_frame"
            goal.pointing_axis.x = 0
            goal.pointing_axis.y = 0
            goal.pointing_axis.z = 1

            goal.target.point.x = x
            goal.target.point.y = y
            goal.target.point.z = z
            goal.max_velocity = velocity
            goal.min_duration = rospy.Duration(1.0)

            ## motion start
            if self.__isMotion == False:
                head_client.send_goal(goal)
                self.__isMotion = True
            rospy.sleep(0.5)
        else:
            print("[Warning] A command is being processed... wait... ")
    def __initParam(self):
        self.__rate = rospy.get_param("~rate", 20)

            # Joint speeds are given in radians per second
        self.__max_joint_speed = rospy.get_param('~max_joint_speed', 0.1)

            # The pan/tilt thresholds indicate what percentage of the image window
            # the ROI needs to be off-center before we make a movement
        self.__pan_threshold = rospy.get_param("~pan_threshold", 0.05)
        self.__tilt_threshold = rospy.get_param("~tilt_threshold", 0.05)

            # The gain_pan and gain_tilt parameter determine how responsive the
            # servo movements are. If these are set too high, oscillation can result.
        self.__gain_pan = rospy.get_param("~gain_pan", 1.0)
        self.__gain_tilt = rospy.get_param("~gain_tilt", 1.0)

        self.__max_pan_angle_radian = rospy.get_param("~max_pan_angle_radian", 1.0)
        self.__max_tilt_angle_radian = rospy.get_param("~max_tilt_angle_radian", 1.0)

    def center(self):
        self.__cmd_pan = 0
        self.__cmd_tilt = 0
        self.move(self.__cmd_pan, self.__cmd_tilt)


    def move(self, pan_rad, tilt_rad, resend=False, debug=False):
        #self.__cmdPub = rospy.Publisher("/gretchen/joint/cmd", Float32MultiArray, queue_size = 0)
        self.__cmdPub = rospy.Publisher("/gretchen/joint/cmd", Float32MultiArray, queue_size = 5)

        print("Goal Pan: \t {:.2f} \t \t Goal Tilt: \t {:.2f}".format(self.__cmd_pan, self.__cmd_tilt))
        head_client = actionlib.SimpleActionClient("/head_controller/absolute_point_head_action",PointHeadAction)
        head_client.wait_for_server()
        head_client.cancel_all_goals()
        if resend == True:
            rospy.sleep(0.4)
            #print "Starting Move Method"
            while True:
                try:
                    image_msg = rospy.wait_for_message("/gretchen/joint/cmd", Float32MultiArray, timeout=0.1)
                    if(image_msg == None):
                        break
                    rospy.sleep(0.2)

                except:
                    pass
                    break

            try:
                image_msg = rospy.wait_for_message("/gretchen/joint/cmd", Float32MultiArray, timeout=0.1)
                if(image_msg == None):
                    rospy.sleep(0.01)
                else:
                    rospy.sleep(0.01)
                    #print type(self.image)
                    #print self.image.shape
                while(image_msg != None):
                    image_msg = rospy.wait_for_message("/gretchen/joint/cmd", Float32MultiArray, timeout=0.1)
                    print("waiting")
                    rospy.sleep(0.2)
            except:
                pass
        cmd = Float32MultiArray()
        self.__cmd_pan = pan_rad
        self.__cmd_tilt = tilt_rad
        cmd.data = [self.__cmd_pan, self.__cmd_tilt]
        while(self.__cmdPub.get_num_connections() < 1):
            rospy.sleep(0.2)
        self.__cmdPub.publish(cmd)

        if(resend == True):
            #Check if moved
            distance = abs(self.__cmd_pan - self.cur_pan_angle) + abs(self.__cmd_tilt - self.cur_tilt_angle)
            counter = 0
            while(distance > 0.05):
                rospy.sleep(0.3)
                distance = abs(self.__cmd_pan - self.cur_pan_angle) + abs(self.__cmd_tilt - self.cur_tilt_angle)
                #print(distance)
                self.__cmdPub.publish(cmd)
                print("Resending command...")
                if counter > 50:
                    break
                counter = counter + 1
        rospy.sleep(1)
        if debug == True:
            self.cur_pan_angle, self.cur_tilt_angle = self.getPosition()
            print("Current Pan: \t {:.2f} \t\t Current Tilt: \t {:.2f}".format(self.cur_pan_angle, self.cur_tilt_angle))


    def down(self, delta=0.1, resend=False):
        self.__cmd_tilt -= delta
        if self.__cmd_tilt < -1.0:
            self.__cmd_tilt = -1.0
        cmd = Float32MultiArray()
        cmd.data = [self.__cmd_pan, self.__cmd_tilt]
        self.move(self.__cmd_pan, self.__cmd_tilt, resend)


    def up(self, delta=0.1, resend=False):
        self.__cmd_tilt += delta
        if self.__cmd_tilt > 1.0:
            self.__cmd_tilt = 1.0
        cmd = Float32MultiArray()
        cmd.data = [self.__cmd_pan, self.__cmd_tilt]
        self.move(self.__cmd_pan, self.__cmd_tilt, resend)


    def left(self, delta=0.1, resend=False):
        self.__cmd_pan += delta
        if self.__cmd_pan > 1.0:
            self.__cmd_pan = 1.0
        cmd = Float32MultiArray()
        cmd.data = [self.__cmd_pan, self.__cmd_tilt]
        self.move(self.__cmd_pan, self.__cmd_tilt, resend)


    def right(self, delta=0.1, resend=False):
        self.__cmd_pan -= delta
        if self.__cmd_pan < -1.0:
            self.__cmd_pan = -1.0
        cmd = Float32MultiArray()
        cmd.data = [self.__cmd_pan, self.__cmd_tilt]
        self.move(self.__cmd_pan, self.__cmd_tilt, resend)

    def __publishCommand(self, x,y):
        cmd = Float32MultiArray()
        cmd.data = [x, y]
        self.__cmdPub.publish(cmd)

    def __actionResultCallback(self, action):
        self.__isMotion = False

    def __jointCallback(self, joint_angles):
        self.cur_pan_angle = joint_angles.data[0]
        self.cur_tilt_angle = joint_angles.data[1]
        #print("Current Pan: {}, Current Tilt: {}".format(self.cur_pan_angle, self.cur_tilt_angle))
        #print(self.__allow_action)
    def getPanAngle(self):
        return self.cur_pan_angle

    def getTiltAngle(self):
        return self.cur_tilt_angle


    def test(self):
        self.__cmd_pan += self.__max_joint_speed
        print(self.__cmd_pan)
        if self.__cmd_pan > 1.0 or self.__cmd_pan < -1.0 :
            self.__max_joint_speed = -1* self.__max_joint_speed
        cmd = Float32MultiArray()
        cmd.data = [self.__cmd_pan, self.__cmd_tilt]
        self.__cmdPub.publish(cmd)
