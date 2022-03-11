from __future__ import print_function

import os
import numpy as np
from absl import app
from absl import flags

from env import Environment
from game import Game
from model_mine import Network
from config import get_config


FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
def load_traffic(trace_file):
    # print('[*]Load loss pattern and delay...', trace_file, delay_file)
    print('[*]Load loss pattern', trace_file)
    f_trace = open(trace_file, 'r')
    loss_pattern = []
    for line in f_trace:
        pattern = [float(v) for v in line.strip().split(' ')]
        line_length = len(pattern)
        loss_pattern.append(pattern)
    f_trace.close()
    loss_pattern = np.array(loss_pattern)
    pattern_shape = loss_pattern.shape #[pattern_cnt, config.state_length]
    pattern_cnt = pattern_shape[0]
    input_state_dims = [pattern_shape[1], 5]
    return loss_pattern

def generate_loss_delay(loss_pattern):
    train_tm_cnt = 11
    state_loss = np.zeros(([train_tm_cnt] + 90), dtype=np.float32) #[N, state_length, tm_history]
    # state_delay = np.zeros(([train_tm_cnt]+ [1] + input_state_dims), dtype=np.float32) #[N, 1, state_length, tm_history]
    for i in range(train_tm_cnt):
        tm_idx = i+tm_history-1
        for h in range(tm_history):
            state_loss[i,:,h] = loss_pattern[tm_idx-h]
            # state_delay[i,:,:,h] = delay[tm_idx-h]
    # state_delay_normalized = state_delay / 150
    state_data = state_loss
    return state_data

def get_state(state_data, tm_idx):
    return state_data[tm_idx]



def sim(config, network, game, trace_file):
    loss_pattern = load_traffic(trace_file)
    state_data = generate_loss_delay(loss_pattern)
    for tm_idx in range(10):# not tm_cnt - history + 1, since the last state is only used to calculate reward
        state = get_state(tm_idx)
        if config.model == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.model == 'pure_policy':
            policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        
        actions = []
        for i in range(config.critical_num + 1):
            actions.append(policy[i*config.action_dim:(i+1)*config.action_dim].argsort()[-1:])

        game.evaluate(config, tm_idx, actions) 

def main(_):
    video_name = 'hall'
    loss_rate = ['0.001', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '0.11', '0.12', '0.13', '0.14', '0.15']
    for lr in loss_rate:
        dataset_path='Emulation_video_test/{}/{}_{}.txt'.format(video_name, video_name, lr)
        flags.DEFINE_string('test_trace_file', dataset_path, 'dataset path')
        
        config = get_config(FLAGS) or FLAGS
        env = Environment(config, is_training=False)
        game = Game(config, env)
        network = Network(config, game.input_state_dims, game.action_dim)

        step = network.restore_ckpt(FLAGS.ckpt)
        if config.model == 'actor_critic':
            learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
        elif config.model == 'pure_policy':
            learning_rate = network.lr_schedule(network.optimizer.iterations.numpy()).numpy()
        print('\nstep %d, learning rate: %f\n'% (step, learning_rate))
    
        sim(config, network, game)


if __name__ == '__main__':
    app.run(main)
