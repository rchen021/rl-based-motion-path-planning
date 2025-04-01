from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        collision_threshold=0.05,
        goal_range=0.42
        ,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.collision_threshold = collision_threshold
        self.object_size = 0.08
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.0, 0.0, 1.0, 1.0]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.0, 0.0, 1.0, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())  # End-effector position
        goal_position = self.goal
        obstacle1_position = np.array(self.sim.get_base_position("object1"))  # Obstacle 1 position
        obstacle2_position = np.array(self.sim.get_base_position("object2"))  # Obstacle 2 position

        return np.concatenate([ee_position, goal_position, obstacle1_position, obstacle2_position])

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

        # Randomize both obstacle positions in every reset
        while True:
            # object1_position and object2_position are randomized within goal range but not colliding with target
            object1_position = self.np_random.uniform(self.goal_range_low[:2], self.goal_range_high[:2])
            object1_position = np.append(object1_position, self.object_size / 2)  # Ensure box stays on the plane

            object2_position = self.np_random.uniform(self.goal_range_low[:2], self.goal_range_high[:2])
            object2_position = np.append(object2_position, self.object_size / 2)  # Ensure box stays on the plane

            # Check that obstacles don't collide with the goal and don't overlap with each other
            if distance(object1_position, self.goal) > self.collision_threshold * 2 and \
               distance(object2_position, self.goal) > self.collision_threshold * 2 and \
               distance(object1_position, object2_position) > self.object_size:
                break

        self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object2_position, np.array([0.0, 0.0, 0.0, 1.0]))


    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        obstacle1_position = np.array(self.sim.get_base_position("object1"))
        obstacle2_position = np.array(self.sim.get_base_position("object2"))
        obstacle1_distance = distance(achieved_goal, obstacle1_position)
        obstacle2_distance = distance(achieved_goal, obstacle2_position)
        
        # If the robot collides with either object, it's a failure
        if obstacle1_distance < self.collision_threshold or obstacle2_distance < self.collision_threshold:
            return np.array(False, dtype=bool)
        
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        obstacle1_position = np.array(self.sim.get_base_position("object1"))
        obstacle2_position = np.array(self.sim.get_base_position("object2"))
        obstacle1_distance = np.linalg.norm(achieved_goal - obstacle1_position)
        obstacle2_distance = np.linalg.norm(achieved_goal - obstacle2_position)

        if self.reward_type != "sparse":
            reward = -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            reward = -d.astype(np.float32)

        if obstacle1_distance < self.collision_threshold:
            reward -= -.1

        if obstacle2_distance < self.collision_threshold:
            reward -= -.1

        return reward.astype(np.float32)

        # d = distance(achieved_goal, desired_goal)
        # if self.reward_type == "sparse":
        #     return -np.array(d > self.distance_threshold, dtype=np.float32)
        # else:
        #     return -d.astype(np.float32)
    
