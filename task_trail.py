import numpy as np
from physics_sim import PhysicsSim
import math
import gym

def sigmoid(x):
    return 1 / (1 + math.exp(-x)) #to be used in reward functions

def barrier(x):
    return sigmoid(-10*(x))

class SampleTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state


class HoverTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=np.array([0., 0., 10.,0.,0.,0.]), init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):#np.array([0., 0., 10.,0.,0.,0.])):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        #TODO: remove 6 to simplify state size to have only vertical?
        self.state_size = self.action_repeat * 12
        self.action_low = 200 #0
        self.action_high = 700 #900
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])#,0.,0.,0.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        max_height = self.sim.upper_bounds[2]
        target_dist = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        accel = np.linalg.norm(self.sim.linear_accel)
        speed = np.linalg.norm(self.sim.v)
        reward = - np.log10((0.2 * np.sum(np.abs(self.sim.pose[:3] - self.target_pos))) + 1.0)
        reward = np.clip(reward, -1.0, 1.0)
        #reward = 1 - 2* (target_dist / 100) - 0.001 * speed
        #if self.sim.pose[2] - self.target_pos[2] > 10:
        #    if self.sim.linear_accel[2] < 0:
        #        reward += 2
        #    else:
        #        reward -= 2
        #if self.sim.pose[2] - self.target_pos[2] < -10:
        #    if self.sim.linear_accel[2] > 0:
        #        reward += 2
        #    else:
        #        reward -= 2 
        #if reward < -10:
        #    reward = -10
        return reward

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        rotor_speeds = [rotor_speed[0]*i for i in [1.,1.,1.,1.]]
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities with all rotors at same speed
            reward += self.get_reward() 
            state_all.append(self.sim.pose)
            state_all.append(self.sim.linear_accel)
            state_all.append(self.sim.angular_accels)
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate((self.sim.pose, self.sim.linear_accel, self.sim.angular_accels))
        state = np.concatenate([state] * self.action_repeat)
        return state


class TakeoffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=np.array([0., 0., 2.,0.,0.,0.]), init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):#np.array([0., 0., 10.,0.,0.,0.])):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        #TODO: remove 6 to simplify state size to have only vertical?
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])#,0.,0.,0.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        max_height = self.sim.upper_bounds[2]
        target_dist = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        accel = np.linalg.norm(self.sim.linear_accel)
        z_vel = self.sim.v[2]
        reward = z_vel
        return reward

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        rotor_speeds = [rotor_speed[0]*i for i in [1.,1.,1.,1.]]
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities with all rotors at same speed
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class MountainCarContinuousTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):#np.array([0., 0., 10.,0.,0.,0.])):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.env = gym.envs.make("MountainCarContinuous-v0")

        self.action_repeat = 1

        self.state_size = self.action_repeat * 2
        self.action_low = self.env.action_space.low[0] #0
        self.action_high = self.env.action_space.high[0] #900
        self.action_size = 1

        
    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        done_flag = 0
        for _ in range(self.action_repeat):
            step_state, step_reward, done, _ = self.env.step(action)
            if done:
                done_flag += 1
            if done_flag < 2: # to allow counting the first reward after done only
                reward += step_reward 
            state_all.append(step_state)
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.env.reset()
        #state = np.concatenate([reset_state] * self.action_repeat) 
        return state