from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config

CFG = Config()


class SyntheticDataset(Dataset):

    AGENT_TYPE_CONFIG: Dict[int, Tuple[float, float, float]] = {
        0: (2.0, 6.0, 0.04),
        1: (0.4, 1.4, 0.22),
        2: (1.0, 3.5, 0.10),
    }

    def __init__(
        self,
        num_scenes: int = 1500,
        obs_len: int = 4,
        pred_len: int = 6,
        max_agents: int = 16,
        dt: float = 0.5,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.max_agents = max_agents
        self.dt = dt
        self._scenes = [self._generate_scene() for _ in range(num_scenes)]

    def _generate_agent(self, agent_type: int) -> np.ndarray:
        v_min, v_max, heading_noise = self.AGENT_TYPE_CONFIG[agent_type]
        speed = np.random.uniform(v_min, v_max)
        heading = np.random.uniform(0, 2 * np.pi)
        x, y = np.random.uniform(-10, 10, 2)
        trajectory = np.zeros((self.total_len, 6), dtype=np.float32)
        for t in range(self.total_len):
            heading += np.random.normal(0, heading_noise)
            speed = max(0.0, speed + np.random.normal(0, 0.2 * speed))
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            trajectory[t] = [x, y, vx, vy, np.cos(heading), np.sin(heading)]
            x += vx * self.dt
            y += vy * self.dt
        return trajectory

    def _compute_visibility(self, trajectory: np.ndarray) -> float:
        last_pos = trajectory[self.obs_len - 1, :2]
        distance = np.sqrt(last_pos[0] ** 2 + last_pos[1] ** 2)
        visibility = np.exp(-distance / 15.0) * np.random.uniform(0.3, 1.0)
        if np.random.rand() < 0.18:
            visibility *= np.random.uniform(0.05, 0.2)
        return float(np.clip(visibility, 0.05, 1.0))

    def _generate_scene(self) -> Dict[str, torch.Tensor]:
        num_agents = np.random.randint(4, self.max_agents + 1)
        past_trajectories = np.zeros((self.obs_len, self.max_agents, 6), dtype=np.float32)
        future_displacements = np.zeros((self.pred_len, self.max_agents, 2), dtype=np.float32)
        visibility_scores = np.zeros((self.max_agents, 1), dtype=np.float32)
        agent_mask = np.zeros(self.max_agents, dtype=bool)
        ref_x = ref_y = 0.0

        for i in range(num_agents):
            agent_type = np.random.choice([0, 0, 0, 1, 1, 2])
            traj = self._generate_agent(agent_type)
            if i == 0:
                ref_x, ref_y = traj[self.obs_len - 1, 0], traj[self.obs_len - 1, 1]
            traj[:, 0] -= ref_x
            traj[:, 1] -= ref_y
            normalised = traj.copy()
            normalised[:, 0] /= CFG.pos_scale
            normalised[:, 1] /= CFG.pos_scale
            normalised[:, 2] /= CFG.vel_scale
            normalised[:, 3] /= CFG.vel_scale
            last_obs_pos = traj[self.obs_len - 1, :2]
            future_disp = (traj[self.obs_len:, :2] - last_obs_pos) / CFG.pos_scale
            past_trajectories[:, i, :] = normalised[:self.obs_len]
            future_displacements[:, i, :] = future_disp
            visibility_scores[i, 0] = self._compute_visibility(traj)
            agent_mask[i] = True

        return {
            'past_trajectory':   torch.from_numpy(past_trajectories),
            'future_trajectory': torch.from_numpy(future_displacements),
            'visibility_mask':   torch.from_numpy(visibility_scores),
            'agent_mask':        torch.from_numpy(agent_mask),
        }

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._scenes[idx]


def get_nuscenes_scene_splits(
    nusc,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    all_scene_names = splits.get('mini_train', []) + splits.get('mini_val', [])
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(all_scene_names))
    num_val_scenes = max(1, int(len(all_scene_names) * val_ratio))
    val_scene_names = {all_scene_names[i] for i in shuffled_indices[:num_val_scenes]}
    train_tokens, val_tokens = [], []
    for scene in nusc.scene:
        token = scene['first_sample_token']
        target_list = val_tokens if scene['name'] in val_scene_names else train_tokens
        while token:
            sample = nusc.get('sample', token)
            target_list.append(token)
            token = sample['next']
    return train_tokens, val_tokens


class NuScenesDataset(Dataset):

    VISIBILITY_SCORE: Dict[int, float] = {1: 0.125, 2: 0.375, 3: 0.625, 4: 0.875}

    def __init__(
        self,
        nusc,
        sample_tokens: List[str],
        obs_len: int = 4,
        pred_len: int = 6,
        max_agents: int = 16,
    ) -> None:
        self.nusc = nusc
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_agents = max_agents
        total_len = obs_len + pred_len
        self.windows: List[List[str]] = []
        for start_token in sample_tokens:
            window = [start_token]
            sample = nusc.get('sample', start_token)
            while sample['next'] and len(window) < total_len:
                window.append(sample['next'])
                sample = nusc.get('sample', sample['next'])
            if len(window) == total_len:
                self.windows.append(window)

    def _get_xy(self, annotation_token: str) -> np.ndarray:
        ann = self.nusc.get('sample_annotation', annotation_token)
        return np.array(ann['translation'][:2], dtype=np.float32)

    def _get_visibility_score(self, annotation_token: str) -> float:
        ann = self.nusc.get('sample_annotation', annotation_token)
        vis_token = ann.get('visibility_token', None)
        if vis_token is None:
            return 0.5
        return self.VISIBILITY_SCORE.get(int(vis_token), 0.5)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        obs_tokens = window[:self.obs_len]
        fut_tokens = window[self.obs_len:]
        instance_to_ann: Dict[str, Dict[str, str]] = {}
        instance_sets = []
        for sample_token in obs_tokens:
            sample = self.nusc.get('sample', sample_token)
            ann_map = {
                self.nusc.get('sample_annotation', a)['instance_token']: a
                for a in sample['anns']
            }
            instance_sets.append(set(ann_map.keys()))
            instance_to_ann[sample_token] = ann_map
        common_instances = list(set.intersection(*instance_sets))[:self.max_agents]
        past_traj = np.zeros((self.obs_len, self.max_agents, 6), dtype=np.float32)
        future_disp = np.zeros((self.pred_len, self.max_agents, 2), dtype=np.float32)
        visibility_scores = np.full((self.max_agents, 1), 0.5, dtype=np.float32)
        agent_mask = np.zeros(self.max_agents, dtype=bool)
        if common_instances:
            ref_pos = self._get_xy(instance_to_ann[obs_tokens[-1]][common_instances[0]])
        else:
            ref_pos = np.zeros(2, dtype=np.float32)
        for agent_idx, instance_token in enumerate(common_instances):
            prev_pos: Optional[np.ndarray] = None
            for t, sample_token in enumerate(obs_tokens):
                ann_token = instance_to_ann[sample_token].get(instance_token)
                if ann_token is None:
                    break
                pos = (self._get_xy(ann_token) - ref_pos) / CFG.pos_scale
                vel = (pos - prev_pos) / 0.5 if prev_pos is not None else np.zeros(2)
                heading = np.arctan2(vel[1], vel[0])
                past_traj[t, agent_idx] = [
                    pos[0], pos[1], vel[0], vel[1], np.cos(heading), np.sin(heading)
                ]
                prev_pos = pos
            visibility_scores[agent_idx, 0] = self._get_visibility_score(
                instance_to_ann[obs_tokens[-1]][instance_token]
            )
            last_obs_pos = past_traj[self.obs_len - 1, agent_idx, :2]
            for t, fut_token in enumerate(fut_tokens):
                fut_sample = self.nusc.get('sample', fut_token)
                fut_anns = {
                    self.nusc.get('sample_annotation', a)['instance_token']: a
                    for a in fut_sample['anns']
                }
                if instance_token in fut_anns:
                    fut_pos = (self._get_xy(fut_anns[instance_token]) - ref_pos) / CFG.pos_scale
                    future_disp[t, agent_idx] = fut_pos - last_obs_pos
            agent_mask[agent_idx] = True
        return {
            'past_trajectory':   torch.from_numpy(past_traj),
            'future_trajectory': torch.from_numpy(future_disp),
            'visibility_mask':   torch.from_numpy(visibility_scores),
            'agent_mask':        torch.from_numpy(agent_mask),
        }
