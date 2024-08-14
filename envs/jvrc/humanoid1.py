import os
import numpy as np
import transforms3d as tf3
import collections

from tasks import stepping_task
from envs.common import mujoco_env
from envs.common import humanoid_interface
from envs.jvrc import robot

class robot:
    def __init__(self):
        self.iteration_count = 0

class HumanoidEnv1(mujoco_env.MujocoEnv):
    def __init__(self):
        sim_dt = 0.0025
        control_dt = 0.025
        self.frame_skip = int(control_dt//sim_dt)

        path_to_xml_out = 'C:\\Users\\cchh\\Desktop\\learn\\LearningHumanoidWalking\\models\\jvrc_mj_description\\xml\\humanoid_torque.xml'
            
        mujoco_env.MujocoEnv.__init__(self, path_to_xml_out, sim_dt, control_dt)

        # set up interface
        self.interface = humanoid_interface.HumanoidInterface(self.model, self.data)
        # set up task
        self.task = stepping_task.SteppingTask(client=self.interface,
                                             dt=control_dt,
                                             neutral_foot_orient=np.array([1, 0, 0, 0]),
                                             root_body='pelvis', head_body='hat_skull',
        )
        # set goal height
        self.task._goal_height_ref = 0.80
        self.task._total_duration = 1.1
        self.task._swing_duration = 0.75
        self.task._stance_duration = 0.35
        # call reset
        self.task.reset()

        self.prev_action = None
        self.prev_torque = None
        self.prev_z = None

        # ndarrays for pd gains
        self.kp = np.array([200] * self.interface.nu())
        self.kd = np.array([20] * self.interface.nu())

        # define indices for action and obs mirror fns
        base_mir_obs = [0.1, -1, 2, -3,              # root orient
                        -4, 5, -6,                   # root ang vel
                        13, -14, -15, 16, -17, 18,   # motor pos [1]
                         7,  -8,  -9, 10, -11, 12,   # motor pos [2]
                        25, -26, -27, 28, -29, 30,   # motor vel [1]
                        19, -20, -21, 22, -23, 24,   # motor vel [2]
        ]
        self.robot = robot()

        # set action space
        self.action_space = np.zeros(self.interface.nu())

        # set observation space
        self.base_obs_len = 84
        self.observation_space = np.zeros(self.base_obs_len)
        
        self.reset_model()

    def get_obs(self):
        # external state
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        ext_state = np.concatenate((clock, [self.task._goal_speed_ref]))

        # internal state
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())

        self.interface.get_lfoot_grf()

        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
        root_ang_vel = qvel[3:6]
        
        l_foot_vec = self.interface.get_lfoot_grf_vec()
        r_foot_vec = self.interface.get_rfoot_grf_vec()

        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()

        robot_state = np.concatenate([
            root_orient,
            root_ang_vel,
            motor_pos,
            motor_vel,
            l_foot_vec[0], l_foot_vec[1],
            r_foot_vec[0], r_foot_vec[1],
        ])
        state = np.concatenate([robot_state, ext_state])
        assert state.shape==(self.base_obs_len,)
        return state.flatten()


    def do_step(self, action):
        filtered_action = action

        if self.prev_action is None:
            self.prev_action = filtered_action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.interface.get_act_joint_torques())
        
        self.prev_z = self.interface.get_object_xpos_by_name("hat_skull", "OBJ_GEOM")[2]

        self.interface.set_pd_gains(self.kp, self.kd)
        self.do_simulation(filtered_action, self.frame_skip)

        self.prev_action = filtered_action
        self.prev_torque = np.asarray(self.interface.get_act_joint_torques())
        return filtered_action

    def do_simulation(self, target, n_frames):
        ratio = self.interface.get_gear_ratios()
        for _ in range(int(n_frames)):
            tau = self.interface.step_pd(target, np.zeros(self.interface.nu()))
            tau = [(i/j) for i,j in zip(tau, ratio)]
            self.interface.set_motor_torque(tau)
            self.interface.step()

    def step(self, a):
        # make one control step
        applied_action = self.do_step(a)

        # compute reward
        self.task.step()
        rewards = self.task.calc_reward(self.prev_torque, self.prev_action, applied_action)
        total_reward = sum([float(i) for i in rewards.values()])

        # check if terminate
        done = self.task.done()

        obs = self.get_obs()
        return obs, total_reward, done, rewards

    def reset_model(self):
        '''
        # dynamics randomization
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0,10)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.2,5)        # actuated joint damping
            self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10) # actuated joint armature
        '''

        c = 0.02
        self.init_qpos = list([0] * self.interface.nq())
        self.init_qvel = list([0] * self.interface.nv())
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv)

        # modify init state acc to task
        self.set_state(
            np.asarray(self.init_qpos),
            np.asarray(self.init_qvel)
        )
        obs = self.get_obs()
        self.task.reset()
        return obs
