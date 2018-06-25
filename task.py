import numpy as np
from physics_sim import PhysicsSim
import math
import gym
import gym.spaces
import gym.envs


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


class TakeOffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=np.array([0., 0., 0.,0.,0.,0.]), init_velocities=None, 
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

        self.state_size = self.action_repeat * 1 # z position only
        
        self.rotor_low = 404 - 404  # 404 is the rotor speed to keep the quadcopter hovering
        self.rotor_high = 404 + 404
        self.rotor_range = self.rotor_high - self.rotor_low

        self.action_low = - 1 # Rescaled actions to pass to the agent
        self.action_high = 1
        self.action_range = self.action_high - self.action_low
        
        self.action_size = 1

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 100.])#,0.,0.,0.])

        self.viewer = None #for rendering

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        target_dist = self.sim.pose[2] - self.target_pos[2]
        reward = 0
        if done:
            if self.sim.time < self.sim.runtime: 
                reward -= 100.0
            else:
                reward += 100 - np.abs(target_dist)
        return reward, done       

    def step(self, rotor_speed):
        """Uses action to obtain next state, reward, done."""
        reward_step = 0
        reward = 0
        state_all = []
        done_flag = 0

        rotor_speeds = [(rotor_speed)*i for i in [1.,1.,1.,1.]]
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities with all rotors at same speed
            done_flag += done
            if done_flag <2: # to allow counting the first reward after done only
                reward_step, done = self.get_reward(done)
                reward += reward_step
            state_all.append(self.sim.pose[2])
        next_state = state_all
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = [self.sim.pose[2]]
        state = np.concatenate([state] * self.action_repeat)
        return state
   
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = 300
        world_height = 300
        scale_z = screen_height/world_height
        scale_x = screen_width/world_width
        coptwidth = 30.0 * scale_x
        coptheight = 5.0 * scale_z

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -coptwidth/2, coptwidth/2, coptheight/2, -coptheight/2
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = 0, screen_width, (self.target_pos[2] * scale_z) + 1, (self.target_pos[2] * scale_z)
            target = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            target.set_color(.8,.2,.2)
            self.viewer.add_geom(target)
            
        if self.sim.pose[2] is None: return None

        x = self.sim.pose
        coptx = x[0] + screen_width/2.0 # MIDDLE OF COPTER
        coptz = x[2] * scale_z
        self.carttrans.set_translation(coptx, coptz)
        #self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


class MountainCarContinuousTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.  """

        # Environment
        self.env = gym.envs.make("MountainCarContinuous-v0")

        self.action_repeat = 3

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
        reset_state = self.env.reset()
        state = np.concatenate([reset_state] * self.action_repeat) 
        return state