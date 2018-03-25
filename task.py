import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
                 init_angle_velocities=None, runtime=5., target_pos=None,
                 target_orientation=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            target_orientation: target/goal (phi, theta, psi) Euler angles in radians
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        # self.action_size = 4
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.target_orientation = target_orientation if target_orientation is not None else np.array([0., 0., 0.])
        self.weight_pos = 2.3
        self.weight_orientation = 0.
        self.weight_velocity = 0.9
        #self.weight_speed_det = 0.0001

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        error_pos = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        error_orientation = np.linalg.norm(self.target_orientation - self.sim.pose[3:])
        #error_velocity = self.sim.v[2]**2
        error_velocity = abs(self.sim.v[2])
        #rotor_speeds_det = np.linalg.det(np.array(rotor_speeds).reshape(2,2))
        # reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum() - .01 * (abs(self.sim.v)).sum()
        reward = 10. - self.weight_pos * error_pos - self.weight_orientation * error_orientation \
                      - self.weight_velocity * error_velocity #- self.weight_speed_det * rotor_speeds_det
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 100.
            self.sim.done = True
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0.
        # pose_all = []
        vp_all = []
        rotor_speeds = np.concatenate([rotor_speeds] * 4)
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds)
            vp_all.append(np.concatenate([self.sim.pose, self.sim.v]))
        next_state = np.concatenate(vp_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v] * self.action_repeat)
        return state
