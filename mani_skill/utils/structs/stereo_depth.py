from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_to_matrix,
)
from mani_skill.utils.structs.base import BaseStruct

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import sapien.sensor
import torch

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose


@dataclass
class StereoDepth(BaseStruct[sapien.sensor.StereoDepthSensor]):
    @classmethod
    def create(self, stereo_depth_cameras: List[sapien.sensor.StereoDepthSensor]):
        pass
