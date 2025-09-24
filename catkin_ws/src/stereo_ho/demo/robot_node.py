#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
import numpy as np

import os
import time

from rospy.numpy_msg import numpy_msg
from std_msgs.msg import UInt64
from std_msgs.msg import String
from frankmocap.msg import Float32

from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg_gripper
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg_gripper
from corsmal_benchmark_s2.msg import Grasps
# Collision IK
import ctypes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState, Grasp
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subject_id', default='0', help='subject id')
parser.add_argument('--config_id', default='0', help='config id')
parser.add_argument('--grasp_width', default=20., type=float, help='grasp width')
FLAGS = parser.parse_args()

class Opt(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int)]

def getPoseDifference(pose1, pose2, norm_pos=False, lx=0.1, ly=0.1, lz=0.1, avg=False, use_centre=True):
    coords = np.array([[-lx, -ly, -lz],
                        [-lx, -ly, lz],
                        [-lx, ly, -lz],
                        [-lx, ly, lz],
                        [lx, -ly, -lz],
                        [lx, -ly, lz],
                        [lx, ly, -lz],
                        [lx, ly, lz]])
    if use_centre:
        coords = np.concatenate((coords, np.zeros((1,3))), axis=0)
    coords = np.concatenate((coords, np.ones((coords.shape[0],1))), axis=1)
    coords1 = np.matmul(pose1, coords.T)[:3,:].T
    coords2 = np.matmul(pose2, coords.T)[:3,:].T

    if norm_pos:
        coords1 = coords1 - coords1[0]
        coords2 = coords2 - coords2[0]

    coords = coords1 - coords2
    if avg:
        diff = np.mean(np.linalg.norm(coords, axis=1))
    else:
        diff = np.sum(np.linalg.norm(coords, axis=1))

    return diff

class MoveRobot(object):
    def __init__(self):
        super(MoveRobot, self).__init__()

        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('MoveRobot', anonymous=True)

        ## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to  the robot:
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This object is an interface to the world surrounding the robot:
        scene = moveit_commander.PlanningSceneInterface()

        ## This interface can be used to plan and execute motions on the Panda:
        group_name = "robot2"
        group = moveit_commander.MoveGroupCommander(group_name)

        ## We create a `DisplayTrajectory`_ publisher which is used later to publish trajectories for RViz to visualize:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.group = group
        self.display_trajectory_publisher = display_trajectory_publisher

        # Moveit settings
        self.group.allow_replanning(True)
        self.group.set_planning_time(1.0)
        self.group.set_max_velocity_scaling_factor(0.3)
        self.group.set_max_acceleration_scaling_factor(0.3)

        # Initial pose
        self.pose_initial = geometry_msgs.msg.Pose()
        self.pose_initial.orientation.x = 0.00440053
        self.pose_initial.orientation.y = -0.00561908
        self.pose_initial.orientation.z = 0.70680023
        self.pose_initial.orientation.w = 0.70737719
        self.pose_initial.position.x = -0.050619
        self.pose_initial.position.y = 0.52567298
        self.pose_initial.position.z =  0.05252385

        # Time for evaluation
        self.timePubB = rospy.Publisher('timeB', UInt64, queue_size=10)
        self.timePubD = rospy.Publisher('timeD', UInt64, queue_size=10)
        self.savedTimeB = None
        self.savedTimeD = None

        # Collision IK
        self.collision_ik_reset()

        # Subscribe to robot joint state
        rospy.Subscriber('/joint_states', JointState, self.callback_js)
        # Subscribe to vision baseline results
        self.hand_pose_received = False
        self.all_grasps_received = False
        self.all_grasps = np.zeros((0,4,4))
        rospy.Subscriber('/hand_mocap/hand_pose_base_repeat', numpy_msg(Float32), self.callbackHandPose)
        rospy.Subscriber('/handover/all_grasps', Grasps, self.callbackAllGrasps)
        self.collect_grasps_iter = 15
        self.min_grasps = 80
        self.max_grasps = 1000

        # Gripper controller
        self.gripper_topic = 'Robotiq2FGripperRobotOutput'
        self.pub_gripper = rospy.Publisher(self.gripper_topic, outputMsg_gripper.Robotiq2FGripper_robot_output, queue_size=10)
        self.gripper_msg = outputMsg_gripper.Robotiq2FGripper_robot_output()

        # Robot controller
        cmd_topic = '/scaled_pos_joint_traj_controller/command'
        self._cmd_pub = rospy.Publisher(cmd_topic, JointTrajectory, queue_size=10)

        # Forward kinematics
        rospy.wait_for_service('compute_fk')
        try:
            self.moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
        except rospy.ServiceException, e:
            rospy.logerror("Service call failed: %s"%e)

        # Robot mode and ids publisher
        self.mode_pub = rospy.Publisher('/recording/robotMode', String, queue_size=10)
        self.subject_id_pub = rospy.Publisher('/recording/subject_id', String, queue_size=10)
        self.config_id_pub = rospy.Publisher('/recording/config_id', String, queue_size=10)
        self.handover_time_pub = rospy.Publisher('/recording/handover_time', String, queue_size=10)

        self.ready = False

    def collision_ik_reset(self):
        # Collision IK
        self.collisionik_lib = ctypes.cdll.LoadLibrary('../relaxed_ik_core/target/debug/librelaxed_ik_lib.so')
        self.collisionik_lib.solve.restype = Opt
        # Initialise
        self.initial_joint_states = self.collision_ik_solve([0.0,0.0,0.0], [0.0,0.0,0.0,1.0], 5000)
        print('Initial joint states: {}'.format(self.initial_joint_states))

    def collision_ik_solve(self, pos, ori, iterations=1000):

        pos_arr = (ctypes.c_double * 3)()
        quat_arr = (ctypes.c_double * 4)()
        pos_arr[0] = pos[0]
        pos_arr[1] = pos[1]
        pos_arr[2] = pos[2]
        quat_arr[0] = ori[0]
        quat_arr[1] = ori[1]
        quat_arr[2] = ori[2]
        quat_arr[3] = ori[3]

        for _ in range(iterations):
            xopt = self.collisionik_lib.solve(pos_arr, len(pos_arr), quat_arr, len(quat_arr))

        joint_states = []
        for i in range(xopt.length):
            jv = float(xopt.data[i])
            if i in [1,3]:
                jv -= 1.570796325
            joint_states.append(jv)

        return joint_states

    def callback_js(self, data):
        self.current_joint_states = [data.position[2], data.position[1], data.position[0], data.position[3], data.position[4], data.position[5]]

    def callbackHandPose(self, data):
        self.hand_pose = np.reshape(data.data, (4,4))
        self.hand_pose_received = True
    
    def callbackAllGrasps(self, data):
        data_all_grasps = np.reshape(data.all_grasps, (-1,4,4))
        data_all_scores = np.reshape(data.all_scores, (-1))

        if self.all_grasps_received == False:
            self.all_grasps = data_all_grasps
            self.all_scores = data_all_scores
            self.all_grasps_received = True
        else:
            self.all_grasps = np.concatenate((self.all_grasps, data_all_grasps), axis=0)
            self.all_scores = np.concatenate((self.all_scores, data_all_scores), axis=0)

    def initGripper(self):
        self.gripper_msg = outputMsg_gripper.Robotiq2FGripper_robot_output()
        self.gripper_msg.rACT = 0
        self.pub_gripper.publish(self.gripper_msg)
        print('Gripper reset')
        rospy.sleep(0.5)
        self.gripper_msg = outputMsg_gripper.Robotiq2FGripper_robot_output()
        self.gripper_msg.rACT = 1
        self.gripper_msg.rGTO = 1
        self.gripper_msg.rATR = 0
        self.gripper_msg.rPR = 0
        self.gripper_msg.rSP = 255
        self.gripper_msg.rFR = 25
        self.pub_gripper.publish(self.gripper_msg)
        print('Gripper activate')
        rospy.sleep(1.0)

    def closeGripper(self, distance):
        #0: open
        #255: closed
        distance *= 1000 # distance to mm from meters
        self.gripper_msg.rPR = max(0, min(255, int(-3*distance+255)))
        #self.gripper_msg.rPR = 255
        self.gripper_msg.rFR = 15
        print('Gripper position {}'.format(self.gripper_msg.rPR))
        self.pub_gripper.publish(self.gripper_msg)
        rospy.sleep(0.5)

    def openGripper(self):
        #0: open
        #255: closed#
        self.gripper_msg.rATR = 1
        self.pub_gripper.publish(self.gripper_msg)
        rospy.sleep(1.2)

    def go_to_joint_state(self, joint_states, duration=0.8, iterations=1, pub_frequency=10.0):

        traj = JointTrajectory()
        traj.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        point = JointTrajectoryPoint()
        point.positions = joint_states
        point.time_from_start = rospy.Duration.from_sec(duration)
        traj.points.append(point)

        for _ in range(iterations):
            self._cmd_pub.publish(traj)
    
        goal_pos, goal_ori, _ = self.compute_fk(joint_states)
        curr_pos, curr_ori, _ = self.compute_fk(self.current_joint_states)
    
        pos_diff = goal_pos - curr_pos
        ori_diff = R.from_quat(goal_ori) * R.from_quat(curr_ori).inv()
        ori_diff = ori_diff.as_euler('xyz', degrees=True)
        
        goal_mat = np.eye(4)
        goal_mat[:3,3] = goal_pos
        goal_mat[:3,:3] = R.from_quat(goal_ori).as_dcm()
        curr_mat = np.eye(4)
        curr_mat[:3,3] = curr_pos
        curr_mat[:3,:3] = R.from_quat(curr_ori).as_dcm()
        diff = getPoseDifference(goal_mat, curr_mat)

        return diff, pos_diff, ori_diff


    def compute_fk(self, joint_states):
        
        fkln = ['robotiq_85_tip_link']
        header = Header(0,rospy.Time.now(),"/world")

        rs = RobotState()
        rs.joint_state.name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        rs.joint_state.position = joint_states

        pose = self.moveit_fk(header, fkln, rs).pose_stamped[0].pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        ori = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        return pos, ori, pose


    def get_collision_ik_pose(self, goal_pos, goal_quat):
        init_mat = np.eye(4)
        init_mat[:3,3] = [self.init_pose.position.x,
                            self.init_pose.position.y,
                            self.init_pose.position.z]
        init_mat[:3,:3] = R.from_quat([self.init_pose.orientation.x,
                                        self.init_pose.orientation.y,
                                        self.init_pose.orientation.z,
                                        self.init_pose.orientation.w]).as_dcm()

        goal_mat = np.eye(4)
        goal_mat[:3,3] = goal_pos
        goal_mat[:3,:3] = R.from_quat(goal_quat).as_dcm()

        collision_ik_goal_mat = np.matmul(goal_mat, np.linalg.inv(init_mat))
        collision_ik_goal_pos = collision_ik_goal_mat[:3,3]
        collision_ik_goal_R = R.from_dcm(collision_ik_goal_mat[:3,:3])
        collision_ik_goal_ori = collision_ik_goal_R.as_quat()

        return collision_ik_goal_pos, collision_ik_goal_ori


    def go_to_collision_ik(self, goal_pos, goal_quat, duration=0.8, steps=20):
        updated_grasp_pose_mat = np.eye(4)
        updated_grasp_pose_mat[:3,3] = goal_pos - self.updated_pos
        updated_grasp_pose_mat[:3,:3] = (R.from_quat(goal_quat) * R.from_quat(self.updated_ori).inv()).as_dcm()

        collision_ik_goal_pos = updated_grasp_pose_mat[:3,3]
        collision_ik_goal_ori = R.from_dcm(updated_grasp_pose_mat[:3,:3]).as_quat()

        # Solve inverse kinematics
        goal_joint_states = self.collision_ik_solve(collision_ik_goal_pos, collision_ik_goal_ori, steps)

        _, _, _ = self.go_to_joint_state(goal_joint_states, duration=duration, iterations=1, pub_frequency=200.0)

        curr_pos, curr_ori, _ = self.compute_fk(self.current_joint_states)
        curr_mat = np.eye(4)
        curr_mat[:3,3] = curr_pos
        curr_mat[:3,:3] = R.from_quat(curr_ori).as_dcm()
        curr_mat = np.matmul(curr_mat, self.robot_delta_mat)
        curr_pos = curr_mat[:3,3]
        curr_ori = R.from_dcm(curr_mat[:3,:3]).as_quat()
    
        pos_diff = goal_pos - curr_pos
        ori_diff = R.from_quat(goal_quat) * R.from_quat(curr_ori).inv()
        ori_diff = ori_diff.as_euler('xyz', degrees=True)
        
        goal_mat = np.eye(4)
        goal_mat[:3,3] = goal_pos
        goal_mat[:3,:3] = R.from_quat(goal_quat).as_dcm()
        curr_mat = np.eye(4)
        curr_mat[:3,3] = curr_pos
        curr_mat[:3,:3] = R.from_quat(curr_ori).as_dcm()
        d = getPoseDifference(goal_mat, curr_mat)

        return d, pos_diff, ori_diff

    def run(self):
        rate = rospy.Rate(200)
        print('Move to initial pose')
        mode = 'init'
        prev_mode = 'init'
        if FLAGS.subject_id == "0" or FLAGS.config_id == "0":
            raise Exception("subject id and config id must be provided")
        
        self.pub_debug_gripper_pose = rospy.Publisher('debug_gripper_pose', PoseStamped, queue_size=10)
        self.pub_debug_grasp_pose = rospy.Publisher('debug_grasp_pose', PoseStamped, queue_size=10)

        # Signal to start recording
        for _ in range(300):
            self.mode_pub.publish(mode)
            self.subject_id_pub.publish(FLAGS.subject_id)
            self.config_id_pub.publish(FLAGS.config_id)
            rate.sleep()

        complete_duration = 2.0
        delivery_duration = 1.0
        prev_best_grasp = None
        prev_hand_pose = None
        approach_cnt = 0
        grasp_change_cooldown = 0
        pick_time = 0.0
        print("========================================================")
        while not rospy.is_shutdown():

            self.mode_pub.publish(mode)
            self.subject_id_pub.publish(FLAGS.subject_id)
            self.config_id_pub.publish(FLAGS.config_id)

            if mode == 'init':
                d, _, _ = self.go_to_joint_state(self.initial_joint_states, duration=4)
                rospy.sleep(4)
                print('Distance to initial pose: {}'.format(d))
                if d < 0.01:
                    
                    mode = 'get_grasps'
                    self.mode_pub.publish(mode)
                    grasp_collection_start_time = rospy.get_time()
                    
                    # Save initial pose
                    _, _, self.init_pose = self.compute_fk(self.initial_joint_states)

                    rotZ = R.from_euler('z', -90.0, degrees=True)
                    self.robot_delta_mat = np.eye(4)
                    self.robot_delta_mat[:3,:3] = rotZ.as_dcm()
                    self.robot_delta_mat[0,3] = -0.22

                    init_pose = np.eye(4)
                    init_pose[:3,3] = [self.init_pose.position.x, self.init_pose.position.y, self.init_pose.position.z]
                    init_pose[:3,:3] = R.from_quat([self.init_pose.orientation.x, self.init_pose.orientation.y, self.init_pose.orientation.z, self.init_pose.orientation.w]).as_dcm()
                    updated_init_pose = np.matmul(init_pose, self.robot_delta_mat)
                    self.updated_pos = updated_init_pose[:3,3]
                    self.updated_ori = R.from_dcm(updated_init_pose[:3,:3]).as_quat()
            else:
                    # Get current robot pose
                    curr_pos, curr_ori, _ = self.compute_fk(self.current_joint_states)
                    curr_mat = np.eye(4)
                    curr_mat[:3,3] = curr_pos
                    curr_mat[:3,:3] = R.from_quat(curr_ori).as_dcm()
                    curr_mat = np.matmul(curr_mat, self.robot_delta_mat)
                    curr_pos = curr_mat[:3,3]
                    curr_ori = R.from_dcm(curr_mat[:3,:3]).as_quat()

            

            if mode == 'get_grasps':
                if not(self.hand_pose_received):
                    print('Waiting for hand pose')
                    continue
                
                if self.all_grasps.shape[0] < self.min_grasps:
                    print('Collecting grasps ... collected {}'.format(self.all_grasps.shape[0]))
                    grasp_collection_time = rospy.get_time() - grasp_collection_start_time
                    if grasp_collection_time > 10:
                        print('Grasp collection time out')
                        mode = 'done'
                        handover_end_time = rospy.get_time()
                        continue
                    continue
                
                # start recording
                for _ in range(300):
                    self.mode_pub.publish("start_recording")
                self.initGripper()
                rospy.sleep(1.7)
                time_start = rospy.get_time()
                handover_start_time = rospy.get_time()

                # Get data
                all_grasps = copy.deepcopy(self.all_grasps)
                all_scores = copy.deepcopy(self.all_scores)
                all_grasps = all_grasps[:self.max_grasps]

                mode = 'approach'
            
            elif mode == 'approach':
                print("========================================================")
                approach_cnt += 1

                time_elapsed = rospy.get_time() - time_start
                if time_elapsed > 10:
                    print("Time out")
                    mode = 'done'
                    handover_end_time = rospy.get_time()
                
                # Find closest grasp to current robot pose
                hand_pose = copy.deepcopy(self.hand_pose)
                hand_pos_y = hand_pose[1,3]
                if hand_pos_y > 1.2:
                    print("Waiting for human ...")
                    continue
                delta_hand = np.eye(4)
                rot_z = R.from_euler('z', -90.0, degrees=True)
                delta_hand[0,3] = -.25
                delta_hand[:3,:3] = rot_z.as_dcm()
                hand_approach = np.matmul(hand_pose, delta_hand)
                hand_pos = hand_approach[:3,3]

                # Get grasps in robot base frame
                all_grasps_curr = copy.deepcopy(all_grasps)
                all_grasps_curr_flipped = copy.deepcopy(all_grasps)
                delta_mat = np.eye(4)
                rotZ = R.from_euler('zx', [180.0,-90.0], degrees=True)
                delta_mat[:3,:3] = rotZ.as_dcm()
                delta_mat_flipped = np.eye(4)
                rotZ = R.from_euler('x', [90.0], degrees=True)
                delta_mat_flipped[:3,:3] = rotZ.as_dcm()

                canonical_grasps = np.concatenate((all_grasps_curr, all_grasps_curr), axis=0)
                canonical_flipped = np.concatenate((np.zeros((len(all_grasps_curr),1)), np.ones((len(all_grasps_curr_flipped),1))), axis=0)
                all_grasps_curr = np.matmul(all_grasps_curr, np.repeat(delta_mat.reshape(1,4,4), len(all_grasps_curr), axis=0))
                all_grasps_curr_flipped = np.matmul(all_grasps_curr_flipped, np.repeat(delta_mat_flipped.reshape(1,4,4), len(all_grasps_curr_flipped), axis=0))
                all_grasps_curr_full = np.concatenate((all_grasps_curr, all_grasps_curr_flipped), axis=0)
                all_grasps_curr_full = np.matmul(np.repeat(hand_pose.reshape(1,4,4), len(all_grasps_curr_full), axis=0), all_grasps_curr_full)
                
                # Publish debug
                gripper_pose = PoseStamped()
                gripper_pose.header.frame_id = 'world'
                gripper_pose.pose.position.x = curr_pos[0]
                gripper_pose.pose.position.y = curr_pos[1]
                gripper_pose.pose.position.z = curr_pos[2]
                gripper_pose.pose.orientation.x = curr_ori[0]
                gripper_pose.pose.orientation.y = curr_ori[1]
                gripper_pose.pose.orientation.z = curr_ori[2]
                gripper_pose.pose.orientation.w = curr_ori[3]
                self.pub_debug_gripper_pose.publish(gripper_pose)
                
                hand_pos, hand_ori = hand_approach[:3,3], R.from_dcm(hand_approach[:3,:3]).as_quat()
            
                # Get grasp closest to robot pose
                diff_list = []
                height_list = []
                rot_list = []
                for grasp in all_grasps_curr_full:
                    grasp_test = copy.deepcopy(grasp)
                    curr_mat_test = copy.deepcopy(curr_mat)
                    robot_diff = getPoseDifference(grasp_test, curr_mat_test)
                    robot_diff_rot = getPoseDifference(grasp_test, curr_mat_test, norm_pos=True)
                    diff_list.append(robot_diff+robot_diff_rot)
                    height_list.append(grasp_test[2,3])
                    rot_list.append(robot_diff_rot)
                top_grasps_rot = np.where(np.array(rot_list) < 1.0)[0]
                top_grasps_h = np.where(np.array(height_list) > -0.1)[0]
                top_grasps_h = np.intersect1d(top_grasps_rot, top_grasps_h)
                all_grasps_curr_full = all_grasps_curr_full[top_grasps_h]
                diff_list = np.array(diff_list)[top_grasps_h]
                top_grasps_d = np.argsort(diff_list)[:min(1, len(diff_list))]
                all_grasps_curr_full = all_grasps_curr_full[top_grasps_d]

                if len(top_grasps_d) == 0:
                    print("Not enough grasps")
                    all_grasps = copy.deepcopy(self.all_grasps)
                    all_scores = copy.deepcopy(self.all_scores)
                    all_grasps = all_grasps[:self.max_grasps]
                    continue
                else:
                    print("Enough grasps")
                
                # Get best grasp based on score
                best_grasp = all_grasps_curr_full[0]
                
                print('Best grasp selected')
                if prev_best_grasp is not None:
                    diff_prev = getPoseDifference(prev_best_grasp, best_grasp, norm_pos=True)
                    print('Difference to previous grasp: {}'.format(diff_prev))
                    if diff_prev > 0.7:
                        print('Difference too large')
                        best_grasp = prev_best_grasp
                    else:
                        print('Difference acceptable')
                        prev_best_grasp = best_grasp
                        grasp_change_cooldown = 5
                else:
                    prev_best_grasp = best_grasp
                
                grasp_change_cooldown -= 1
                        
                # Publish debug
                best_grasp_pos = best_grasp[:3,3]
                best_grasp_ori = R.from_dcm(best_grasp[:3,:3]).as_quat()
                grasp_pose = PoseStamped()
                grasp_pose.header.frame_id = 'world'
                grasp_pose.pose.position.x = best_grasp_pos[0]
                grasp_pose.pose.position.y = best_grasp_pos[1]
                grasp_pose.pose.position.z = best_grasp_pos[2]
                grasp_pose.pose.orientation.x = best_grasp_ori[0]
                grasp_pose.pose.orientation.y = best_grasp_ori[1]
                grasp_pose.pose.orientation.z = best_grasp_ori[2]
                grasp_pose.pose.orientation.w = best_grasp_ori[3]
                self.pub_debug_grasp_pose.publish(grasp_pose)

                # Get approach pose
                grasp_pose_mat = np.eye(4)
                grasp_pose_mat[:3,3] = best_grasp[:3,3]
                grasp_pose_mat[:3,:3] = best_grasp[:3,:3]
                updated_grasp_pose_mat = grasp_pose_mat

                approach_mat = np.eye(4)
                approach_mat[1,3] = -0.25
                approach_mat[2,3] = -0.0
                approach_mat = np.matmul(updated_grasp_pose_mat, approach_mat)

                approach_grasp_pos = approach_mat[:3,3]
                if approach_grasp_pos[2] < -0.1:
                    approach_grasp_pos[2] = -0.1
                
                # Get speed based on distance to approach pose
                distance_to_approach = getPoseDifference(approach_mat, curr_mat, avg=True, use_centre=False)
                speed = 0.20
                steps = 300
                approach_grasp_ori = R.from_dcm(approach_mat[:3,:3]).as_quat()
                
                time_elapsed = rospy.get_time() - time_start
                duration = distance_to_approach / speed
                duration = max(0.5, duration)
                if time_elapsed > 10:
                    print("Time out")
                    mode = 'pick'
                print('Duration: {}'.format(duration))
                d, pos_diff, ori_diff = self.go_to_collision_ik(approach_grasp_pos, approach_grasp_ori, duration=duration, steps=steps)
                approach_mat = np.eye(4)
                approach_mat[:3,3] = approach_grasp_pos
                approach_mat[:3,:3] = R.from_quat(approach_grasp_ori).as_dcm()

                pos_diff_done = all([abs(p) < 0.005 for p in pos_diff])
                ori_diff_done = all([abs(o) < 10.0 for o in ori_diff])

                if d < 0.2 and approach_cnt > 5:
                    print('Reached approach pose')
                    mode = 'pick'

            elif mode == 'pick':
                if pick_time == 0.0:
                    pick_time = rospy.get_time()
                    # get current robot joint states
                    pre_grasp_joint_states = self.current_joint_states
                pick_mat = np.eye(4)
                pick_mat[1,3] = 0.24
                pick_mat[2,3] = 0.0
                pick_mat = np.matmul(approach_mat, pick_mat)

                goal_pos = np.array([pick_mat[0,3], pick_mat[1,3], pick_mat[2,3]])
                goal_quat = R.from_dcm(pick_mat[:3,:3]).as_quat()

                distance_to_goal = np.linalg.norm(goal_pos - curr_pos)
                speed = 0.20
                duration = distance_to_goal / speed

                d, pos_diff, ori_diff = self.go_to_collision_ik(goal_pos, goal_quat, duration=duration, steps=300)

                pos_diff_done = all([abs(p) < 0.005 for p in pos_diff])
                ori_diff_done = all([abs(o) < 2.0 for o in ori_diff])

                time_elapsed = rospy.get_time() - time_start
                if time_elapsed > 10:
                    print("Time out")
                    self.closeGripper(float(FLAGS.grasp_width)/1000.0)
                    mode = 'retract'

                if d < 0.3:
                    print('Reached grasp pose')
                    self.closeGripper(float(FLAGS.grasp_width)/1000.0)
                    mode = 'retract'


            elif mode == 'retract':

                _, _, _ = self.go_to_joint_state(pre_grasp_joint_states, duration=2)
                rospy.sleep(2)
                print('Reached retract pose')
                mode = 'delivery'
                    


            elif mode == 'delivery':
                delivery_pose = copy.deepcopy(updated_init_pose)
                delivery_collision_ik_delta = np.eye(4)
                delivery_pos = delivery_pose[:3,3]
                delivery_pos[1] += 0.35
                delivery_pos[2] -= 0.15
                delivery_ori = R.from_dcm(delivery_pose[:3,:3]).as_quat()

                distance_to_goal = np.linalg.norm(delivery_pos - curr_pos)
                speed = 0.25
                duration = distance_to_goal / speed

                d, pos_diff, ori_diff = self.go_to_collision_ik(delivery_pos, delivery_ori, duration=duration)

                if d < 0.5:
                    print('Reached delivery pose')
                    self.openGripper()
                    mode = 'completing1'
                    handover_end_time = rospy.get_time()


            elif mode == 'completing1':
                completing_pose = copy.deepcopy(delivery_pose)
                completing_pos = completing_pose[:3,3]
                completing_pos[2] -= 0.08
                completing_ori = R.from_dcm(completing_pose[:3,:3]).as_quat()

                distance_to_goal = np.linalg.norm(completing_pos - curr_pos)
                speed = 0.2
                duration = distance_to_goal / speed

                d, pos_diff, ori_diff = self.go_to_collision_ik(completing_pos, completing_ori, duration=duration)

                if d < 0.3:
                    print('Completed')
                    mode = 'completing2'
            
            elif mode == 'completing2':
                completing_pose = copy.deepcopy(delivery_pose)
                completing_pos = completing_pose[:3,3]
                completing_pos[2] -= 0.08
                completing_pos[1] -= 0.15
                completing_ori = R.from_dcm(completing_pose[:3,:3]).as_quat()

                distance_to_goal = np.linalg.norm(completing_pos - curr_pos)
                speed = 0.25
                duration = distance_to_goal / speed                

                d, pos_diff, ori_diff = self.go_to_collision_ik(completing_pos, completing_ori, duration=duration)

                if d < 0.2:
                    print('Completed')
                    mode = 'done'
                    


            elif mode == 'done':
                d, _, _ = self.go_to_joint_state(self.initial_joint_states, duration=complete_duration)
                if d < 3.0:
                    complete_duration = 2.0 - ((3.0-d)/3.0)*1.8
                
                if d < 0.1:
                    print("Moved to home position!")
                    self.openGripper()
                    mode = 'stop_recording'
                    handovertime = str(handover_end_time - handover_start_time)
                    for _ in range(300):
                        self.mode_pub.publish(mode)
                        self.handover_time_pub.publish(handovertime)

                    break

            if mode != prev_mode:
                print("========================================================")
                print("Mode: {}".format(mode))
                prev_mode = mode

            
            self.timePubB.publish(self.savedTimeB)
            self.timePubD.publish(self.savedTimeD)


        return
        

if __name__ == '__main__':
    robot = MoveRobot()
    robot.run()




