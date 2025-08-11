#!/usr/bin/env python2
import time
import rospy
import numpy as np
import torch
import os
import pickle

from moveit_msgs.msg import Grasp
from graspnet.srv import EstimateGrasps, EstimateGraspsResponse

from scipy.spatial.transform import Rotation as R

# 6DOFGraspNet
import grasp_estimator
import mayavi.mlab as mlab
from utils.visualization_utils import *
import mayavi.mlab as mlab
from utils import utils
import argparse

def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=20)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=512)
    parser.add_argument('--num_grasp_samples', type=int, default=150)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=150,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    return parser


def getSafeGrasps(grasps, scores, pc_h):
    filtered_grasps = []
    filtered_scores = []
    y_min, y_max = -0.01, 0.01
    x_min, x_max = -0.035, 0.035
    z_min, z_max = -0.15, 0.20

    pc_h = np.concatenate([pc_h, np.ones((pc_h.shape[0], 1))], axis=1)
    for i in range(grasps.shape[0]):
        grasp = grasps[i]
        score = scores[i]

        pc_h_grasp = np.matmul(np.linalg.inv(grasp), pc_h.T).T[:,:3]
        # Check if pointcloud of hand pc_h is in bounding box of the grasp bbox_grasp
        in_collision = False
        for j in range(pc_h_grasp.shape[0]):
            if pc_h_grasp[j][0] > x_min and pc_h_grasp[j][0] < x_max and pc_h_grasp[j][1] > y_min and pc_h_grasp[j][1] < y_max and pc_h_grasp[j][2] > z_min and pc_h_grasp[j][2] < z_max:
                in_collision = True
                break
        if not in_collision:
            filtered_grasps.append(grasp)
            filtered_scores.append(score)
    filtered_grasps = np.array(filtered_grasps)
    filtered_scores = np.array(filtered_scores)

    print("Removed {} grasps out of {}".format(grasps.shape[0] - filtered_grasps.shape[0], grasps.shape[0]))
    return filtered_grasps, filtered_scores


class GraspEstimationRosNode:
    def __init__(self):
        self.idx = 0
        self.seg_output = None
        self.hand_pose = None
        self.img_size = 256
        # Points for coordinate system
        coords = np.array([[0, 0, 0],
                            [0.1, 0, 0],
                            [0, 0.1, 0],
                            [0, 0, 0.1]])
        self.coords = torch.from_numpy(coords).float().unsqueeze(0)

        self.pc_h = np.zeros((1,3))
        self.pc_o = np.zeros((1,3))

        rospy.init_node('GraspEstimation', anonymous=True, disable_signals=True)
        s = rospy.Service('grasp_estimation', EstimateGrasps, self.estimate_grasp)

        self.pc_h_received = False
        self.pc_o_received = False

        # 6DOFGraspNet
        parser = make_parser()
        args = parser.parse_args()
        grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
        grasp_sampler_args.is_train = False
        grasp_evaluator_args = utils.read_checkpoint_args(
            args.grasp_evaluator_folder)
        grasp_evaluator_args.continue_train = True
        self.estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                                    grasp_evaluator_args, args)

        # Read camera calibration
        calib_path = '../../../calib'
        target2base_f = os.path.join(calib_path, 'robot2', 'C_target2base.pkl')
        cam2target_f = os.path.join(calib_path, 'cam2', 'C_baseline.pkl')

        with open(target2base_f, 'rb') as f:
            target2base = pickle.load(f)
        with open(cam2target_f, 'rb') as f:
            cam2target_data = pickle.load(f)
            cam2target_rvec = cam2target_data['extrinsic']['rgb']['rvec']
            cam2target_tvec = cam2target_data['extrinsic']['rgb']['tvec'].reshape(3)
            cam2target = np.eye(4)
            cam2target[:3, :3] = cam2target_rvec
            cam2target[:3, 3] = cam2target_tvec
            cam2target = np.linalg.inv(cam2target)
        
        print("Loaded camera calibration ...")

        self.cam2base = np.matmul(target2base, cam2target)

        rospy.spin()
        
    def estimate_grasp(self, req):
        drawDebug = False
        grasp = Grasp()

        # Read request
        pc_h = np.reshape(req.pc_h, (-1,3)).astype(np.float32)
        pc_o = np.reshape(req.pc_o, (-1,3)).astype(np.float32)

        grasp.grasp_quality = 0.0
        all_start = time.time()
        print('============================')
        pc_colors = np.zeros((pc_o.shape[0], 3), dtype=np.uint8)
        pc_colors[:, :] = [255, 0, 0]
        start_time = time.time()
        with torch.no_grad():
            generated_grasps, generated_scores = self.estimator.generate_and_refine_grasps_safe(
                pc_o, pc_h)
        print("Time for grasp generation: ", time.time()-start_time)
        generated_grasps = np.array(generated_grasps)
        generated_scores = np.array(generated_scores)

        if generated_grasps.shape[0] == 0:
            print("No grasps generated")
            response = EstimateGraspsResponse()
            response.all_grasps = np.array([]).astype(np.float32)
            response.all_scores = np.array([]).astype(np.float32)
            return response
            
        if drawDebug:
            pc_vis = np.concatenate([pc_o, pc_h], axis=0)
            pc_colors = np.zeros((pc_vis.shape[0], 3), dtype=np.uint8)
            pc_colors[:pc_o.shape[0]] = [255, 0, 0]
            
            mlab.figure(bgcolor=(1, 1, 1))
            # Plot coordinate xyz
            mlab.plot3d([0, 0.1], [0, 0], [0, 0], color=(1, 0, 0), tube_radius=0.001)
            mlab.plot3d([0, 0], [0, 0.1], [0, 0], color=(0, 1, 0), tube_radius=0.001)
            mlab.plot3d([0, 0], [0, 0], [0, 0.1], color=(0, 0, 1), tube_radius=0.001)
            draw_scene(
                pc_vis,
                pc_color=pc_colors,
                grasps=generated_grasps,
                grasp_scores=generated_scores,
            )
            mlab.show()

        # Check if grasp is in collision with hand
        start_time = time.time()
        filtered_grasps, filtered_scores = getSafeGrasps(generated_grasps, generated_scores, pc_h)
        print("Filtered grasps: ", filtered_grasps.shape[0])
        print("Time for collision check: ", time.time()-start_time)

        if drawDebug:
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(
                pc_vis,
                pc_color=pc_colors,
                grasps=filtered_grasps,
                grasp_scores=filtered_scores,
            )
            mlab.show()

        response = EstimateGraspsResponse()
        response.all_grasps = filtered_grasps.reshape(-1).astype(np.float32)
        response.all_scores = filtered_scores.reshape(-1).astype(np.float32)

        print("Time for all: ", time.time()-all_start)

        return response


if __name__ == '__main__':
    node = GraspEstimationRosNode()