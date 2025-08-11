# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time

# From IHOI
from ihoi.nnutils.hand_utils import ManopthWrapper
from ihoi.nnutils.handmocap import get_handmocap_predictor, process_mocap_predictions, get_handmocap_detector
from ihoi.nnutils.geom_utils import se3_to_matrix
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import Transform3d, Rotate, Translate, Scale

def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    #Set up input data (images or webcam)
    input_type, input_data = demo_utils.setup_input(args)
 
    assert args.out_dir is not None, "Please specify output dir to store the results"
    cur_frame = args.start_frame
    video_frame = 0

    while True:
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        
        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        # bbox detection
        if load_bbox:
            body_pose_list = None
            raw_hand_bboxes = None
        elif args.crop_type == 'hand_crop':
            # hand already cropped, thererore, no need for detection
            img_h, img_w = img_original_bgr.shape[:2]
            body_pose_list = None
            raw_hand_bboxes = None
            hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
        else:            
            # Input images has other body part or hand not cropped.
            # Use hand detection model & body detector for hand detection
            assert args.crop_type == 'no_crop'
            detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
            body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
        
        # save the obtained body & hand bbox to json file
        if args.save_bbox_output:
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(hand_bbox_list) < 1:
            print(f"No hand deteced: {image_path}")
            continue
    
        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # IHOI integration
        print("IHOI integration")
        object_mask = np.ones_like(img_original_bgr[..., 0]) * 255
        hand_wrapper = ManopthWrapper(mano_path="extra_data/mano/models").to('cpu')
        ihoi_data = process_mocap_predictions(pred_output_list, img_original_bgr, hand_wrapper, mask=object_mask)
        cropped_image = ihoi_data['image'].squeeze(0).numpy().transpose(1, 2, 0)
        cropped_image = (cropped_image/2.0 + 0.5) * 255.0
        cropped_image = np.ascontiguousarray(cropped_image, dtype=np.uint8)

        # Get world to camera transform
        cTh = ihoi_data['cTh']
        trans = se3_to_matrix(cTh)
        trans = Transform3d(matrix=trans.transpose(1,2))

        # Points for coordinate system
        coords = np.array([[0, 0, 0],
                            [0, 0, 0.1],
                            [0, 0.1, 0],
                            [0.1, 0, 0]])
        coords = torch.from_numpy(coords).float().unsqueeze(0)
        norm_coords = trans.transform_points(coords)

        # Points for sdf grid
        SDF_MULTIPLIER = 1.0
        BBOX_TOPLEFT = np.array([0.2, 0.16, 0.25])
        BBOX_BOTTOMRIGHT = np.array([-0.32, -0.36, -0.27])
        BBOX_ORIG_X = SDF_MULTIPLIER * ((0.2+(-0.32))/2.0)
        BBOX_ORIG_Y = SDF_MULTIPLIER * ((0.16+(-0.36))/2.0)
        BBOX_ORIG_Z = SDF_MULTIPLIER * ((0.25+(-0.27))/2.0)
        BBOX_SIZE_X = SDF_MULTIPLIER * (0.2-(-0.32))
        BBOX_SIZE_Y = SDF_MULTIPLIER * (0.16-(-0.36))
        BBOX_SIZE_Z = SDF_MULTIPLIER * (0.25-(-0.27))
        sdf_grid = np.meshgrid(np.linspace(BBOX_TOPLEFT[0], BBOX_BOTTOMRIGHT[0], 9),
                                np.linspace(BBOX_TOPLEFT[1], BBOX_BOTTOMRIGHT[1], 9),
                                np.linspace(BBOX_TOPLEFT[2], BBOX_BOTTOMRIGHT[2], 9))
        print(sdf_grid)
        print(sdf_grid[0].shape)
        sdf_grid = np.stack(sdf_grid, axis=-1)
        print(sdf_grid.shape)
        sdf_grid = sdf_grid.reshape(-1, 3)
        print(sdf_grid.shape)
        sdf_grid = torch.from_numpy(sdf_grid).float().unsqueeze(0)
        print(sdf_grid.shape)
        norm_sdf_grid = trans.transform_points(sdf_grid)
        
        # Project to image space
        camera = PerspectiveCameras(ihoi_data['cam_f'], ihoi_data['cam_p'], device='cpu', image_size=(224, 224))
        coords_2d = camera.transform_points_screen(norm_coords)
        coords_2d = coords_2d[0][:,:2].numpy()
        sdf_grid_2d = camera.transform_points_screen(norm_sdf_grid)
        sdf_grid_2d = sdf_grid_2d[0][:,:2].numpy()

        # Draw coords on cropped image
        for i in range(4):
            colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            if i != 0:
                start = (int(224-coords_2d[0][0]), int(224-coords_2d[0][1]))
                end = (int(224-coords_2d[i][0]), int(224-coords_2d[i][1]))
                cv2.line(cropped_image, start, end,  colours[i-1], 2)

        # Draw sdf grid on cropped image
        for i in range(sdf_grid_2d.shape[0]):
            cv2.circle(cropped_image, (int(224-sdf_grid_2d[i][0]), int(224-sdf_grid_2d[i][1])), 1, (0, 0, 255), -1)
                
        # Show cropped image
        cv2.imshow("cropped", cropped_image)
        
        # Save cropped image
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, cropped_image, folder_name='cropped')

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize
        res_img = visualizer.visualize(
            img_original_bgr, 
            pred_mesh_list = pred_mesh_list, 
            hand_bbox_list = hand_bbox_list)

        # show result in the screen
        crop_center = ihoi_data['crop_center']
        hoi_bbox = ihoi_data['hoi_bbox']
        print(hoi_bbox)
        scale = (hoi_bbox[2] - hoi_bbox[0])/224.
        print(crop_center)
        cv2.circle(res_img, (int(crop_center[0]), int(crop_center[1])), 5, (0, 0, 255), -1)
        for i in range(4):
            colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            if i != 0:
                start = np.array([224-coords_2d[0][0], 224-coords_2d[0][1]])
                end = np.array([224-coords_2d[i][0], 224-coords_2d[i][1]])
                start = start - 224/2
                end = end - 224/2
                start = start * scale
                end = end * scale
                start = start + crop_center
                end = end + crop_center
                start = (int(start[0])+640, int(start[1]))
                end = (int(end[0])+640, int(end[1]))
                cv2.line(res_img, start, end,  colours[i-1], 2)
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)
        print(res_img.shape)
        # save the image (we can make an option here)
        if args.out_dir is not None:
            demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'hand'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        print(f"Processed : {image_path}")
        
    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    # When everything done, release the capture
    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()

  
def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
