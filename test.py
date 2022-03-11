from __future__ import print_function

import os
import numpy as np
from absl import app
from absl import flags

from env import Environment
from game import Game
from model_mine import Network
from config import get_config
import time
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')

def sim(config, network, game):
    sum = 0
    for tm_idx in range(game.tm_cnt - game.tm_history):# not tm_cnt - history + 1, since the last state is only used to calculate reward
        time1 = time.time()
        state = game.get_state(tm_idx)
        if config.model == 'actor_critic':
            policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.model == 'pure_policy':
            policy = network.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        
        actions = []
        for i in range(config.critical_num + 1):
            actions.append(policy[i*config.action_dim:(i+1)*config.action_dim].argsort()[-1:])
        time2 = time.time()
        # print(time2 - time1)
        sum += time2 - time1
        game.evaluate(config, tm_idx, actions) 
    print(sum)

def main(_):
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
