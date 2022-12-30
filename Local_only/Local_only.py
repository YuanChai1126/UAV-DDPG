import math
import random

import numpy as np


class UAVEnv(object):
    height = ground_length = ground_width = 100  
    sum_task_size = 60 * 1048576  
    loc_uav = [50, 50]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6  
    p_noisy_los = 10 ** (-13)  
    p_noisy_nlos = 10 ** (-11)  
    flight_speed = 50. 
    f_ue = 6e8 
    f_uav = 12e8  
    r = 10 ** (-27)  
    s = 1000 
    p_uplink = 0.1  
    # alpha0 = -30  
    alpha0 = 1e-5  
    T = 200  
    delta_t = 5  
    slot_num = int(T / delta_t)  
    m_uav = 9.65  
    e_battery_uav = 500000  

    #################### ues ####################
    M = 4  
    block_flag_list = np.random.randint(0, 2, M)  
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  
    # task_list = np.random.randint(1048576, 2097153, M)    
    task_list = np.random.randint(1572864, 2097153, M)  
    
   
    loc_ue_trans_pro = np.array([[.6, .1, .1, .1, .1],
                                 [.6, .1, .1, .1, .1],
                                 [.6, .1, .1, .1, .1],
                                 [.6, .1, .1, .1, .1]])

    action_bound = [-1, 1]  
    action_dim = 4 
    state_dim = 4 + M * 4  # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag

    def __init__(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.start_state = np.append(self.e_battery_uav, self.loc_uav)
        self.start_state = np.append(self.start_state, self.sum_task_size)
        self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
        self.start_state = np.append(self.start_state, self.task_list)
        self.start_state = np.append(self.start_state, self.block_flag_list)
        self.state = self.start_state

    def reset(self):
        self.reset_env()
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    def reset_env(self):
        self.sum_task_size = 100 * 1048576  
        self.e_battery_uav = 500000 
        self.loc_uav = [50, 50]
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])  
        self.reset_step()

    def reset_step(self):
        # self.task_list = np.random.randint(1572864, 2097153, self.M)  
        # self.task_list = np.random.randint(2097152, 2621441, self.M)  
        self.task_list = np.random.randint(2621440, 3145729, self.M)  
        # self.task_list = np.random.randint(3145728, 3670017, self.M)  
        # self.task_list = np.random.randint(3670016, 4194305, self.M)  
        self.block_flag_list = np.random.randint(0, 2, self.M)  

    def _get_obs(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self.state

    def step(self): 
        step_redo = False
        is_terminal = False
        ue_id = np.random.randint(0, self.M)

        theta = 0  
        offloading_ratio = 0  
        task_size = self.task_list[ue_id]
        block_flag = self.block_flag_list[ue_id]

        
        dis_fly = 0  
       
        e_fly = (dis_fly / (self.delta_t * 0.5)) ** 2 * self.m_uav * (
                self.delta_t * 0.5) * 0.5  # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning

       
        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        loc_uav_after_fly_x = self.loc_uav[0] + dx_uav
        loc_uav_after_fly_y = self.loc_uav[1] + dy_uav

        
        t_server = offloading_ratio * task_size / (self.f_uav / self.s)  
        e_server = self.r * self.f_uav ** 3 * t_server  

        if self.sum_task_size == 0:  
            is_terminal = True
            # file_name = 'output.txt'
            # with open(file_name, 'a') as file_obj:
            #     file_obj.write("\n======== This episode is done ========")  
            reward = 0
        elif self.sum_task_size - self.task_list[ue_id] < 0:  
            self.task_list = np.ones(self.M) * self.sum_task_size
            reward = 0
            step_redo = True
        elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > self.ground_width or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > self.ground_length:  
            reward = -100
            step_redo = True
        elif self.e_battery_uav < e_fly:  
            reward = -100
        elif self.e_battery_uav - e_fly < e_server:  
            reward = -100
        else:  
            delay = self.com_delay(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   offloading_ratio, task_size, block_flag)  
            reward = delay
            
            self.e_battery_uav = self.e_battery_uav - e_fly - e_server  
            self.sum_task_size -= self.task_list[ue_id]  
            for i in range(self.M):  
                tmp = np.random.rand()
                if 0.6 < tmp <= 0.7:
                    self.loc_ue_list[i] += [0, 1]
                elif 0.7 < tmp <= 0.8:
                    self.loc_ue_list[i] += [1, 0]
                elif 0.8 < tmp <= 0.9:
                    self.loc_ue_list[i] += [0, -1]
                elif 0.9 < tmp <= 1:
                    self.loc_ue_list[i] += [-1, 0]
                else:
                    self.loc_ue_list[i] += [0, 0]
                np.clip(self.loc_ue_list[i], 0, 100)
            # self.task_list = np.random.randint(1048576, 2097153, self.M)  
            self.reset_step()

            
            # file_name = 'output.txt'
            # # file_name = 'output_' + str(len(self.UE_loc_list)) + 'UE_DDPG.txt'
            # with open(file_name, 'a') as file_obj:
            #     file_obj.write("\nUE-" + '{:d}'.format(ue_id) + ", task size: " + '{:d}'.format(
            #         int(task_size)) + ", offloading ratio:" + '{:.2f}'.format(offloading_ratio))
            #     file_obj.write("\ndelay:" + '{:.2f}'.format(delay))
            #     file_obj.write("\nUAV hover loc:" + "[" + '{:.2f}'.format(loc_uav_after_fly_x) +
            #                    ', ' + '{:.2f}'.format(loc_uav_after_fly_y) + ']')  

        return reward, is_terminal, step_redo

    
    def com_delay(self, loc_ue, loc_uav, offloading_ratio, task_size, block_flag):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)  
        t_tr = offloading_ratio * task_size / trans_rate  
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)  
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s) 
        return max([t_tr + t_edge_com, t_local_com])


def diff_bandwidth():
    for k in range(10):
        delays_list = []
        for j in range(1, 11, 1):
            env = UAVEnv()
            env.reset()
            env.B = j * 10 ** 6  
            costs = 0
            i = 0
            while i < env.slot_num:
                delay, is_terminal, step_redo = env.step()
                costs += delay
                if step_redo:
                    continue
                if is_terminal or i == env.slot_num - 1:
                    delays_list.append(eval("{:.4f}".format(costs)))
                    break
                i = i + 1
        print(np.array(delays_list))


def diff_task_size():
    delays_list = []
    for k in range(10):
        
        env = UAVEnv()
        env.reset()
        costs = 0
        i = 0
        while i < env.slot_num:
            delay, is_terminal, step_redo = env.step()
            costs += delay
            if step_redo:
                continue
            if is_terminal or i == env.slot_num - 1:
                delays_list.append(eval("{:.4f}".format(costs)))
                break
            i = i + 1
    print(np.mean(delays_list))

def diff_f_ue():
    delays_list = []
    for k in range(20):
        
        env = UAVEnv()
        env.reset()
        costs = 0
        i = 0
        while i < env.slot_num:
            delay, is_terminal, step_redo = env.step()
            costs += delay
            if step_redo:
                continue
            if is_terminal or i == env.slot_num - 1:
                delays_list.append(costs)
                break
            i = i + 1
    print(np.around(np.mean(delays_list), 4))

if __name__ == '__main__':
    diff_f_ue()
    # diff_bandwidth()
    # diff_task_size()

'''
different bandwidth：
[104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576 104.8576]

'''