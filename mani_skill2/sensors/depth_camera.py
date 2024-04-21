from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import sapien.core as sapien
from gymnasium import spaces
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

from mani_skill2.utils.sapien_utils import get_entity_by_name

from .camera import Camera, CameraConfig

import sys
import torch
import torch.nn.functional as F

class StereoDepthCameraConfig(CameraConfig):
    def __init__(self, *args, min_depth: float = 0.05, trt_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_depth = min_depth
        self.trt_dict = trt_dict

    @property
    def rgb_resolution(self):
        return (self.width, self.height)

    @property
    def rgb_intrinsic(self):
        fy = (self.height / 2) / np.tan(self.fov / 2)
        return np.array([[fy, 0, self.width / 2], [0, fy, self.height / 2], [0, 0, 1]])

    @classmethod
    def fromCameraConfig(cls, cfg: CameraConfig):
        return cls(**cfg.__dict__)


class StereoDepthCamera(Camera):
    def __init__(
        self,
        camera_cfg: StereoDepthCameraConfig,
        scene: sapien.Scene,
        renderer_type: str,
        articulation: sapien.Articulation = None,
    ):
        self.camera_cfg = camera_cfg
        self.trt_dict = camera_cfg.trt_dict
        assert renderer_type == "sapien", renderer_type
        self.renderer_type = renderer_type

        actor_uid = camera_cfg.actor_uid
        if actor_uid is None:
            self.actor = None
        else:
            if articulation is None:
                self.actor = get_entity_by_name(scene.get_all_actors(), actor_uid)
            else:
                self.actor = get_entity_by_name(articulation.get_links(), actor_uid)
            if self.actor is None:
                raise RuntimeError(f"Mount actor ({actor_uid}) is not found")

        # Add camera
        sensor_config = StereoDepthSensorConfig()
        sensor_config.rgb_resolution = camera_cfg.rgb_resolution

        sensor_config.rgb_intrinsic = camera_cfg.rgb_intrinsic
        sensor_config.min_depth = camera_cfg.min_depth
        if self.actor is None:
            self.camera = StereoDepthSensor(
                camera_cfg.uid, scene, sensor_config, mount=self.actor
            )
            self.camera.set_pose(camera_cfg.pose)
        else:
            self.camera = StereoDepthSensor(
                camera_cfg.uid,
                scene,
                sensor_config,
                mount=self.actor,
                pose=camera_cfg.pose,
            )

        if camera_cfg.hide_link:
            self.actor.hide_visual()

        # Filter texture names according to renderer type if necessary (legacy for Kuafu)
        self.texture_names = camera_cfg.texture_names

    def get_images(self, take_picture=False):
        # trt_dict = {"model": stereo_trt(), "focal": 480., "baseline":0.1}
        """Get (raw) images from the camera."""
        if take_picture:
            self.take_picture()

        if self.renderer_type == "client":
            return {}

        images = {}

        for name in self.texture_names:
            
            if name == "Color":
                image = self.camera._cam_rgb.get_float_texture("Color")
            elif name == "depth":
                self.camera.compute_depth()
                image = self.camera.get_depth()[..., None]
            elif name == "Position":
                self.camera.compute_depth()
                position = self.camera._cam_rgb.get_float_texture("Position")
                if self.trt_dict is None:
                    depth = self.camera.get_depth()
                    position[..., 2] = -depth
                else:
                    trt_stereo_model = self.trt_dict["model"]

                    left_tensor = self.camera._cam_ir_l.get_color_rgba()
                    right_tensor = self.camera._cam_ir_r.get_color_rgba()
                    left_tensor = left_tensor[:,:,:3].transpose([2,0,1])
                    right_tensor = right_tensor[:,:,:3].transpose([2,0,1])

                    left_tensor = torch.Tensor(left_tensor)[None,...]
                    right_tensor = torch.Tensor(right_tensor)[None,...]

                    
                    left_tensor = F.interpolate(left_tensor, size=(384, 768)).numpy()
                    right_tensor = F.interpolate(right_tensor, size=(384, 768)).numpy()

                    pred = trt_stereo_model.infer(left_tensor, right_tensor)

                    pred = torch.Tensor(pred)[None,None,...]
                    pred = F.interpolate(pred, size=(128,128)).numpy()[0,0]

                    depth_pred = (self.trt_dict["focal"] * self.trt_dict["baseline"])/pred
                    # depth_pred = self.camera.compute_depth_by_model()
                    position[..., 2] = -depth_pred
                image = position
            elif name == "Segmentation":
                image = self.camera._cam_rgb.get_uint32_texture("Segmentation")
            else:
                raise NotImplementedError(name)
            images[name] = image
        return images

    def get_params(self):
        """Get camera parameters."""
        return dict(
            extrinsic_cv=self.camera._cam_rgb.get_extrinsic_matrix(),
            cam2world_gl=self.camera._cam_rgb.get_model_matrix(),
            intrinsic_cv=self.camera._cam_rgb.get_intrinsic_matrix(),
        )

    @property
    def observation_space(self) -> spaces.Dict:
        obs_spaces = OrderedDict()
        width, height = self.camera._cam_rgb.width, self.camera._cam_rgb.height
        for name in self.texture_names:
            if name == "Color":
                obs_spaces[name] = spaces.Box(
                    low=0, high=1, shape=(height, width, 4), dtype=np.float32
                )
            elif name == "Position":
                obs_spaces[name] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(height, width, 4),
                    dtype=np.float32,
                )
            elif name == "Segmentation":
                obs_spaces[name] = spaces.Box(
                    low=np.iinfo(np.uint32).min,
                    high=np.iinfo(np.uint32).max,
                    shape=(height, width, 4),
                    dtype=np.uint32,
                )
            else:
                raise NotImplementedError(name)
        return spaces.Dict(obs_spaces)
