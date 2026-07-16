#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from pathlib import Path
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, is_technicolor_scene
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import find_best_iteration_from_log
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
import torch
import numpy as np
from utils.compressed_loader import load_gaussians_from_compressed


class Scene:
    gaussians : GaussianModel
    def __init__(
        self,
        args : ModelParams,
        gaussians : GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        load_coarse=False,
        use_compressed=False,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.criterion =  torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]).cuda()) 
        self.use_compressed = use_compressed

        # load_iteration = 13000
        if load_iteration:
            if load_iteration == -1:
                # self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                self.loaded_iter, _ = find_best_iteration_from_log(os.path.join(self.model_path, "log_metrics.txt"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        technicolor_sparse = os.path.join(args.source_path, "colmap", "dense", "workspace", "sparse")

        if is_technicolor_scene(args.source_path):
            if not os.path.exists(technicolor_sparse):
                raise FileNotFoundError(
                    f"Technicolor scene detected at {args.source_path} but dense workspace is missing.\n"
                    f"Run `python scripts/pre_technicolor.py --videopath {args.source_path}` first."
                )
            duration = getattr(args, "technicolor_duration", None)
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args.source_path, args.images, args.eval, duration)
            dataset_type="Technicolor"
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))

        if hasattr(self.gaussians, "_deformation"):
            try:
                self.gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
            except AttributeError:
                pass

        if self.loaded_iter:
            iter_dir = os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}")
            if self.use_compressed:
                load_gaussians_from_compressed(self.gaussians, Path(self.model_path), self.loaded_iter)
            else:
                self.gaussians.load_ply(os.path.join(iter_dir, "point_cloud.ply"))
            self.gaussians.load_model(iter_dir)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_hash(point_cloud_path)



    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    
    def getVideoCameras(self, scale=1.0):
        return self.video_camera
