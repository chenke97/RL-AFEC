from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from RS_decode_video import decode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# print(tf.__version__)
tf.get_logger().setLevel('ERROR')
# from video_quality import load_model
from get_vq import get_video_quality
#import matplotlib.pyplot as plt

UTILIZATION_COST = False            #True: utilization cost, False: delay
OBJ_EPSILON = 1e-12
_NEG_INF_FP32 = -1e9
NORMALIZED_TM_BASE = 1e6

class Game(object):
    def __init__(self, config, env, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)
        self.tm_history = config.tm_history
        self.critical_num = config.critical_num
        self.predict_interval = config.predict_interval
        self.max_moves = config.max_moves
        
        self.action_dim = env.action_dim
        self.actions_pool = config.actions_pool
            
 
        if config.softmax_temperature:
            self.softmax_temperature = np.sqrt(self.action_dim)
        else:
            self.softmax_temperature = 1
        
        self.normalized_reward = config.normalized_reward
        self.data_dir = env.data_dir
        
        self.loss_pattern = env.loss_pattern
        
        self.input_state_dims = env.input_state_dims # [state_length, tm_history]
        self.tm_cnt = env.tm_cnt

        self.tm_indexes = np.arange(self.tm_history-1, self.tm_cnt)
        self.train_tm_cnt = len(self.tm_indexes)
        
        self.generate_loss_delay()
        self.baseline = {}
        # self.video_quality_model = self.vmaf_predict_model()
    
    def vmaf_predict_model(self):
        model = keras.Sequential([
            layers.Dense(90, activation='relu', input_shape=[90]),
            #     layers.Dense(90, activation='relu', input_shape=[90]),
            layers.Dense(90, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(0.001)

        model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
        #     model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        wights_file = 'Weights-494--5.61865.hdf5' # choose the best checkpoint 
        model.load_weights(wights_file)
        return model
    def get_state(self, tm_idx):
        
        return self.state_data[tm_idx] # N, 2, 30, 5
    
    def generate_loss_delay(self):
        self.state_loss = np.zeros(([self.train_tm_cnt] + self.input_state_dims), dtype=np.float32) #[N, state_length, tm_history]
        # self.state_delay = np.zeros(([self.train_tm_cnt]+ [1] + self.input_state_dims), dtype=np.float32) #[N, 1, state_length, tm_history]
        for i in range(self.train_tm_cnt):
            tm_idx = i+self.tm_history-1
            for h in range(self.tm_history):
                self.state_loss[i,:,h] = self.loss_pattern[tm_idx-h]
                # self.state_delay[i,:,:,h] = self.delay[tm_idx-h]
        # self.state_delay_normalized = self.state_delay / 150
        self.state_data = self.state_loss
        

    # def get_video_quality(loss_pattern):
    #     vmaf = self.NN_model.predict(loss_pattern)
        
    #     return vmaf
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
        
    def penalty(self, vmaf):
        if vmaf < 40:
            return vmaf/(1 + 0.1*(80 - vmaf))
        elif vmaf < 60:
            return 0.6*vmaf - 16
        else:
            return 3*vmaf - 160
    
    def reward(self, tm_idx, actions):
        alpha = 0.5
        vmaf_target = np.array([[80.0]])
        bw_lowerbound = np.array([[30.0]]) # 
        loss_pattern_current = self.state_loss[tm_idx+1,:,0]# the current action is for the next state
        critical_frames_num = self.critical_num
        percentage = []
        for i in range(self.critical_num + 1):
            percentage.append(self.actions_pool[actions[i][0]]) # actions: [array([0]), array([3]), ...]
        loss_pattern_after, bw_waste = decode(loss_pattern_current.squeeze(), percentage, critical_frames_num)
        loss_pattern_after = loss_pattern_after.reshape((1, 90))
        # vmaf = get_video_quality(loss_pattern_after)
        
        vmaf = self.vmaf_predict_model().predict(loss_pattern_after)
        # vmaf = self.video_quality_model.predict(loss_pattern_after)
        if vmaf > vmaf_target:
            r = 5*(vmaf_target/80) - alpha * (bw_waste / 1500) / bw_lowerbound
        else:
            # r = (self.penalty(vmaf)/vmaf_target)*5 - alpha * (bw_waste / 1500) / bw_lowerbound
            r = - vmaf_target / (self.penalty(vmaf) + 1)
        
        #return r
        return vmaf, bw_waste, loss_pattern_after #for testing

    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]
        
        #print(reward, (total_v/cnt))

        return reward - (total_v/cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        max_x = np.max(x)
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum()

    def evaluate(self, config, tm_idx, actions):
        file_name = config.test_trace_file.split('/')[-1]
        f = open("eval_results_"+file_name, "a+")
        f1 = open("loss_pattern_after_"+file_name, "a+")
        
        # reward = self.reward(tm_idx, actions)
        vmaf, bw_waste, loss_pattern_after = self.reward(tm_idx, actions)
        # f.write(str(tm_idx+1)+" "+str(actions)+" "+str(reward) + " " + str(pkt_loss_after_recovery) + "\n")
        vmaf = np.asscalar(vmaf)
        for i in range(self.critical_num + 1):
            actions[i] = np.asscalar(actions[i])
        f.write(str(tm_idx+1)+" "+str(actions)+" "+str(vmaf)+" "+str(bw_waste)+ "\n")
        f1.write(str(loss_pattern_after.squeeze()[:30]) + "\n")
        f.close()
        f1.close()
