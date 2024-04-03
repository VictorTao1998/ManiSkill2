from .sequential_task import SequentialTaskEnv
from .planner import (
    TaskPlan,
    Subtask,
    PlaceSubtask,
    SubtaskConfig,
    PlaceSubtaskConfig,
)

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_raw_multiply
from mani_skill import ASSET_DIR

import torch
import torch.random
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import sapien
import sapien.physx as physx

from typing import Any, Dict, List
from pathlib import Path
from tqdm import tqdm
import itertools
import pickle
import copy


PLACE_OBS_EXTRA_KEYS = [
    "tcp_pose_wrt_base",
    "obj_pose_wrt_base",
    "goal_pos_wrt_base",
    "is_grasped",
]


@register_env("PlaceSequentialTask-v0", max_episode_steps=200)
class PlaceSequentialTaskEnv(SequentialTaskEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    # TODO (arth): add locomotion, open fridge, close fridge
    pick_cfg = None
    place_cfg = PlaceSubtaskConfig(
        horizon=200,
        obj_goal_thresh=0.15,
        ee_rest_thresh=0.05,
    )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        # spawn randomization
        randomize_arm=False,
        randomize_base=False,
        randomize_loc=False,
        # additional spawn randomization, shouldn't need to change
        spawn_loc_radius=2,
        # colliison tracking
        robot_force_mult=0,
        robot_force_penalty_min=0,
        robot_cumulative_force_limit=torch.inf,
        # additional randomization
        goal_randomization=False,
        **kwargs,
    ):

        # NOTE (arth): task plan length and order checking left to SequentialTaskEnv
        tp0 = task_plans[0]
        assert len(tp0) == 1 and isinstance(
            tp0[0], PlaceSubtask
        ), "Task plans for Place training must be one PlaceSubtask long"

        # randomization vals
        self.randomize_arm = randomize_arm
        self.randomize_base = randomize_base
        self.randomize_loc = randomize_loc
        self.spawn_loc_radius = spawn_loc_radius

        # force reward hparams
        self.robot_force_mult = robot_force_mult
        self.robot_force_penalty_min = robot_force_penalty_min
        self.robot_cumulative_force_limit = robot_cumulative_force_limit

        # additional randomization
        self.goal_randomization = goal_randomization

        super().__init__(*args, robot_uids=robot_uids, task_plans=task_plans, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # COLLISION TRACKING
    # -------------------------------------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
        return super().reset(*args, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # INIT RANDOMIZATION
    # -------------------------------------------------------------------------------------------------
    # TODO (arth): maybe check that obj won't fall when noise is added
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    def _get_navigable_spawn_positions_with_rots_and_dists(self, center_x, center_y):
        # NOTE (arth): this is all unbatched, should be called wtih DEFAULT obj spawn pos
        center = torch.tensor([center_x, center_y])
        pts = torch.tensor(self.scene_builder.navigable_positions)
        pts_wrt_center = pts - center

        dists = torch.norm(pts_wrt_center, dim=1)
        in_circle = dists <= self.spawn_loc_radius
        pts, pts_wrt_center, dists = (
            pts[in_circle],
            pts_wrt_center[in_circle],
            dists[in_circle],
        )

        rots = (
            torch.sign(pts_wrt_center[:, 1])
            * torch.arccos(pts_wrt_center[:, 0] / dists)
            + torch.pi
        )
        rots %= 2 * torch.pi

        return torch.hstack([pts, rots.unsqueeze(-1)]), dists

    def _after_reconfigure(self, options):
        with torch.device(self.device):
            super()._after_reconfigure(options)

            robot_pick_success_states_fp: Path = (
                Path(ASSET_DIR)
                / "robot_success_states"
                / self.robot_uids
                / "pick"
                / f"{self.base_task_plans[0][0].obj_id}.pkl"
            )
            assert (
                robot_pick_success_states_fp.exists()
            ), f"Need {str(robot_pick_success_states_fp)} to load robot start states"

            with open(robot_pick_success_states_fp, "rb") as f:
                robot_pick_success_states = pickle.load(f)
            self.init_robot_qpos = torch.tensor(robot_pick_success_states["robot_qpos"])
            self.init_obj_pose_wrt_base = torch.tensor(
                robot_pick_success_states["obj_pose_wrt_base"]
            )

            self.scene_builder.initialize(torch.arange(self.num_envs))

            if physx.is_gpu_enabled():
                self._scene._gpu_apply_all()
                self._scene.px.gpu_update_articulation_kinematics()
                self._scene._gpu_fetch_all()

            # links and entities for force tracking
            force_rew_ignore_links = [
                self.agent.finger1_link,
                self.agent.finger2_link,
                self.agent.tcp,
            ]
            self.force_articulation_link_ids = [
                link.name
                for link in self.agent.robot.get_links()
                if link not in force_rew_ignore_links
            ]

            goal = self.subtask_goals[0]

            spawn_loc_rots = []
            spawn_dists = []
            for env_idx in range(self.num_envs):
                center = goal.pose.p[env_idx, :2]
                slr, dists = self._get_navigable_spawn_positions_with_rots_and_dists(
                    center[0], center[1]
                )
                spawn_loc_rots.append(slr)
                spawn_dists.append(dists)

            num_spawn_loc_rots = torch.tensor([len(slr) for slr in spawn_loc_rots])
            spawn_loc_rots = pad_sequence(
                spawn_loc_rots, batch_first=True, padding_value=0
            ).transpose(1, 0)
            spawn_dists = pad_sequence(
                spawn_dists, batch_first=True, padding_value=0
            ).transpose(1, 0)

            qpos = torch.tensor(
                self.agent.RESTING_QPOS[..., None]
                .repeat(self.num_envs, axis=-1)
                .transpose(1, 0)
            ).float()
            accept_spawn_loc_rots = [[] for _ in range(self.num_envs)]
            accept_dists = [[] for _ in range(self.num_envs)]
            bounding_box_corners = [
                torch.tensor([dx, dy, 0])
                for dx, dy in itertools.product([0.1, -0.1], [0.1, -0.1])
            ]
            for slr_num, (slrs, dists) in tqdm(
                enumerate(zip(spawn_loc_rots, spawn_dists)),
                total=spawn_loc_rots.size(0),
            ):

                slrs_within_range = slr_num < num_spawn_loc_rots
                robot_force = torch.zeros(self.num_envs)

                for shift in bounding_box_corners:
                    shifted_slrs = slrs + shift

                    self.agent.controller.reset()
                    qpos[..., 2] = shifted_slrs[..., 2]
                    self.agent.reset(qpos)

                    # ad-hoc use z-rot dim a z-height dim, set using default setting
                    shifted_slrs[..., 2] = self.agent.robot.pose.p[..., 2]
                    self.agent.robot.set_pose(
                        Pose.create_from_pq(p=shifted_slrs.float())
                    )

                    if physx.is_gpu_enabled():
                        self._scene._gpu_apply_all()
                        self._scene.px.gpu_update_articulation_kinematics()
                        self._scene._gpu_fetch_all()

                    self._scene.step()

                    robot_force += (
                        self.agent.robot.get_net_contact_forces(
                            self.force_articulation_link_ids
                        )
                        .norm(dim=-1)
                        .sum(dim=-1)
                    )

                for i in torch.where(slrs_within_range & (robot_force < 1e-3))[0]:
                    accept_spawn_loc_rots[i].append(slrs[i].cpu().numpy().tolist())
                    accept_dists[i].append(dists[i].item())

            self.num_spawn_loc_rots = torch.tensor(
                [len(x) for x in accept_spawn_loc_rots]
            )
            self.spawn_loc_rots = pad_sequence(
                [torch.tensor(x) for x in accept_spawn_loc_rots],
                batch_first=True,
                padding_value=0,
            )

            self.closest_spawn_loc_rots = torch.stack(
                [
                    self.spawn_loc_rots[i][torch.argmin(torch.tensor(x))]
                    for i, x in enumerate(accept_dists)
                ],
                dim=0,
            )

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            super()._initialize_episode(env_idx, options)

            b = len(env_idx)

            if self.goal_randomization:
                xyz = torch.zeros((b, 3))
                xyz[..., :2] = torch.rand((b, 2)) * 0.2
                xyz += torch.tensor(self.task_plan[0].goal_pos)
                self.subtask_goals[0].set_pose(Pose.create_from_pq(p=xyz))

            self.resting_qpos = torch.tensor(self.agent.RESTING_QPOS[3:-2])

            pick_init_idxs = torch.randint(len(self.init_robot_qpos), (b,))
            qpos = self.init_robot_qpos[pick_init_idxs]
            qpos[..., :3] = 0
            obj_pose_wrt_base = self.init_obj_pose_wrt_base[pick_init_idxs]

            if self.randomize_loc:
                idxs = torch.tensor(
                    [
                        torch.randint(max_idx.item(), (1,))
                        for max_idx in self.num_spawn_loc_rots
                    ]
                )
                loc_rot = self.spawn_loc_rots[torch.arange(self.num_envs), idxs]
            else:
                loc_rot = self.closest_spawn_loc_rots
            robot_pos = self.agent.robot.pose.p
            robot_pos[..., :2] = loc_rot[..., :2]
            self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))

            qpos[..., 2] = loc_rot[..., 2]
            if self.randomize_base:
                # base pos
                robot_pos = self.agent.robot.pose.p
                robot_pos[..., :2] += torch.clamp(
                    torch.normal(0, 0.04, (b, len(robot_pos[0, :2]))), -0.075, 0.075
                ).to(self.device)
                self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos))
                # base rot
                qpos[..., 2:3] += torch.clamp(
                    torch.normal(0, 0.25, (b, len(qpos[0, 2:3]))), -0.5, 0.5
                ).to(self.device)

            self.agent.reset(qpos)
            if physx.is_gpu_enabled():
                self._scene._gpu_apply_all()
                self._scene.px.gpu_update_articulation_kinematics()
                self._scene._gpu_fetch_all()

            if self.randomize_arm:
                old_tcp_pose = self.agent.tcp_pose

                qpos[..., 5:6] += torch.clamp(
                    torch.normal(0, 0.05, (b, len(qpos[0, 5:6]))), -0.1, 0.1
                ).to(self.device)
                qpos[..., 7:-2] += torch.clamp(
                    torch.normal(0, 0.05, (b, len(qpos[0, 7:-2]))), -0.1, 0.1
                ).to(self.device)

                if physx.is_gpu_enabled():
                    self._scene._gpu_apply_all()
                    self._scene.px.gpu_update_articulation_kinematics()
                    self._scene._gpu_fetch_all()
                new_tcp_pose = self.agent.tcp_pose

                pose_transformation = old_tcp_pose.inv() * new_tcp_pose
                self.subtask_objs[0].set_pose(
                    self.agent.base_link.pose * obj_pose_wrt_base * pose_transformation
                )
            else:
                self.subtask_objs[0].set_pose(
                    self.agent.base_link.pose * obj_pose_wrt_base
                )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------
    # Remove irrelevant obs for place task from state dict
    # -------------------------------------------------------------------------------------------------

    def _get_obs_state_dict(self, info: Dict):
        state_dict = super()._get_obs_state_dict(info)

        extra_state_dict_keys = list(state_dict["extra"])
        for key in extra_state_dict_keys:
            if key not in PLACE_OBS_EXTRA_KEYS:
                state_dict["extra"].pop(key, None)

        return state_dict

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): evaluate() function here to support continuous task wrapper on cpu sim
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        with torch.device(self.device):
            infos = super().evaluate()

            # set to zero in case we use continuous task wrapper in cpu sim
            #   this way, if the termination signal is ignored, env will
            #   still reevaluate success each step
            self.subtask_pointer = torch.zeros_like(self.subtask_pointer)

            robot_force = (
                self.agent.robot.get_net_contact_forces(
                    self.force_articulation_link_ids
                )
                .norm(dim=-1)
                .sum(dim=-1)
            )
            self.robot_cumulative_force += robot_force

            infos.update(
                robot_force=robot_force,
                robot_cumulative_force=self.robot_cumulative_force,
            )

            return infos

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)

            obj_pos = self.subtask_objs[0].pose.p
            goal_pos = self.subtask_goals[0].pose.p
            rest_pos = self.ee_rest_world_pose.p
            tcp_pos = self.agent.tcp_pose.p

            # NOTE (arth): reward "steps" are as follows:
            #       - reaching_reward
            #       - is_grasped_reward
            #       - if grasped and not at goal
            #           - obj to goal reward
            #       - if at goal
            #           - rest reward
            #       - if at rest
            #           - static reward
            #       - success_reward
            # ---------------------------------------------------
            # CONDITION CHECKERS
            # ---------------------------------------------------

            obj_to_goal_dist = torch.norm(obj_pos - goal_pos, dim=1)
            obj_at_goal = obj_to_goal_dist <= self.place_cfg.obj_goal_thresh
            obj_at_goal_reward = torch.zeros_like(reward[obj_at_goal])

            obj_not_at_goal = ~obj_at_goal
            obj_not_at_goal_reward = torch.zeros_like(reward[obj_not_at_goal])

            ee_to_rest_dist = torch.norm(tcp_pos - rest_pos, dim=1)
            ee_rest = obj_at_goal & (ee_to_rest_dist <= self.place_cfg.ee_rest_thresh)
            ee_rest_reward = torch.zeros_like(reward[ee_rest])

            # ---------------------------------------------------

            # penalty for ee jittering too much
            ee_vel = self.agent.tcp.linear_velocity
            ee_still_rew = 1 - torch.tanh(torch.norm(ee_vel, dim=1) / 5)
            reward += ee_still_rew

            # penalty for object moving too much
            obj_vel = torch.norm(
                self.subtask_objs[0].linear_velocity, dim=1
            ) + torch.norm(self.subtask_objs[0].angular_velocity, dim=1)
            obj_still_rew = 3 * (1 - torch.tanh(obj_vel / 5))
            reward += obj_still_rew

            # success reward
            success_rew = 3 * info["success"]
            reward += success_rew

            # encourage arm and torso in "resting" orientation
            arm_to_resting_diff = torch.norm(
                self.agent.robot.qpos[..., 3:-2] - self.resting_qpos,
                dim=1,
            )
            arm_resting_orientation_rew = 2 * (1 - torch.tanh(arm_to_resting_diff))
            reward += arm_resting_orientation_rew

            # penalty for torso moving up and down too much
            tqvel_z = self.agent.robot.qvel[..., 3]
            torso_not_moving_rew = 1 - torch.tanh(5 * torch.abs(tqvel_z))
            reward += torso_not_moving_rew

            # ---------------------------------------------------------------
            # colliisions
            step_no_col_rew = 1 - torch.tanh(
                3
                * (
                    torch.clamp(
                        self.robot_force_mult * info["robot_force"],
                        min=self.robot_force_penalty_min,
                    )
                    - self.robot_force_penalty_min
                )
            )
            reward += step_no_col_rew

            # cumulative collision penalty
            cum_col_under_thresh_rew = (
                info["robot_cumulative_force"] < self.robot_cumulative_force_limit
            ).float()
            reward += cum_col_under_thresh_rew
            # ---------------------------------------------------------------

            if torch.any(obj_not_at_goal):
                # ee places obj at goal (instead of throwing)
                obj_not_at_goal_reward += 2 * info["is_grasped"][obj_not_at_goal]

                # obj place reward
                place_rew = 5 * (1 - torch.tanh(obj_to_goal_dist[obj_not_at_goal]))
                obj_not_at_goal_reward += place_rew

                x = torch.zeros_like(reward)
                x[obj_not_at_goal] = place_rew

                # rew for ee over goal
                ee_over_goal_rew = 1 - torch.tanh(
                    5
                    * torch.norm(
                        tcp_pos[..., :2][obj_not_at_goal]
                        - goal_pos[..., :2][obj_not_at_goal],
                        dim=1,
                    )
                )
                obj_not_at_goal_reward += ee_over_goal_rew

                x = torch.zeros_like(reward)
                x[obj_not_at_goal] = ee_over_goal_rew

            if torch.any(obj_at_goal):
                # add prev step max rew
                obj_at_goal_reward += 8

                # obj_left_at_goal
                obj_at_goal_reward += 2 * ~info["is_grasped"][obj_at_goal]

                # rest reward
                rest_rew = 5 * (1 - torch.tanh(3 * ee_to_rest_dist[obj_at_goal]))
                obj_at_goal_reward += rest_rew

                x = torch.zeros_like(reward)
                x[obj_at_goal] = rest_rew

                # additional encourage arm and torso in "resting" orientation
                more_arm_resting_orientation_rew = 2 * (
                    1 - torch.tanh(arm_to_resting_diff[obj_at_goal])
                )
                obj_at_goal_reward += more_arm_resting_orientation_rew

                x = torch.zeros_like(reward).float()
                x[obj_at_goal] = more_arm_resting_orientation_rew.float()

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][obj_at_goal]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                obj_at_goal_reward += base_still_rew

                x = torch.zeros_like(reward)
                x[obj_at_goal] = base_still_rew

            if torch.any(ee_rest):
                ee_rest_reward += 2

                qvel = self.agent.robot.qvel[..., :-2][ee_rest]
                static_rew = 1 - torch.tanh(torch.norm(qvel, dim=1))
                ee_rest_reward += static_rew

                x = torch.zeros_like(reward)
                x[ee_rest] = static_rew

                # penalty for base moving or rotating too much
                bqvel = self.agent.robot.qvel[..., :3][ee_rest]
                base_still_rew = 1 - torch.tanh(torch.norm(bqvel, dim=1))
                ee_rest_reward += base_still_rew

                x = torch.zeros_like(reward)
                x[ee_rest] = base_still_rew

            # add rewards to specific envs
            reward[obj_not_at_goal] += obj_not_at_goal_reward
            reward[obj_at_goal] += obj_at_goal_reward
            reward[ee_rest] += ee_rest_reward

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 34.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------
