from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import inspect
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# from layers import transformer

class Network():
    def __init__(self, config, input_dims, action_dim, mask=None, master=True):
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.critical_num = config.critical_num
        self.attention_mask = mask
        self.model_name = config.version+'-'\
                            +config.model+'_'\
                            +config.trace_file

        if config.model == 'actor_critic':
            if config.graph_extractor == 'Conv':
                self.create_actor_critic_model(config)
            elif config.graph_extractor == 'Transformer':
                self.create_actor_critic_attention_model(config)
        elif config.model == 'pure_policy':
            if config.graph_extractor == 'Conv':
                self.create_policy_model(config)
            elif config.graph_extractor == 'Transformer':
                self.create_policy_attention_model(config)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                config.initial_learning_rate,
                config.learning_rate_decay_step,
                config.learning_rate_decay_rate,
                staircase=True)

        if config.optimizer == 'RMSprop':
            if config.model == 'actor_critic':
                self.actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
                self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
            elif config.model == 'pure_policy':
                self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
        elif config.optimizer == 'Adam':
            if config.model == 'actor_critic':
                self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
                self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
            elif config.model == 'pure_policy':
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        if master:
            if config.model == 'actor_critic':
                self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), actor_optimizer=self.actor_optimizer, critic_optimizer=self.critic_optimizer, model=self.model)
            elif config.model == 'pure_policy':
                self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.policy_model)
            self.ckpt_dir = './tf_ckpts/'+self.model_name
            self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=config.max_to_keep)
            self.writer = tf.compat.v2.summary.create_file_writer('./logs/%s' % self.model_name)
            #self.save_hyperparams(config)

    def create_actor_critic_model(self, config):
        tf.keras.backend.set_floatx('float32')
        # inputs = tf.keras.Input(shape=(2, self.input_dims[0], self.input_dims[1]))
        inputs = tf.keras.Input(shape=(self.input_dims[0], self.input_dims[1]))# not including the batch size
        # inputs_loss_pattern = inputs[:,0,:,:]
        inputs_loss_pattern = inputs
        # inputs_delay = inputs[:,1,:,:]
        # tf.reshape(inputs_loss_pattern, [1, self.input_dims[0], self.input_dims[1]])
        # tf.reshape(inputs_delay, [1, self.input_dims[0], self.input_dims[1]])
        # Actor
        Conv1D_1_loss_pattern = tf.keras.layers.Conv1D(config.Conv1D_out, 1, strides=1, padding='valid', use_bias=not config.batch_norm)
        # Conv1D_1_delay = tf.keras.layers.Conv1D(config.Conv1D_out, 1, strides=1, padding='valid', use_bias=not config.batch_norm)
        
        x_1_loss_pattern = Conv1D_1_loss_pattern(inputs_loss_pattern)
        # x_1_delay = Conv1D_1_delay(inputs_delay)

        if config.batch_norm:
            x_1_loss_pattern = tf.keras.layers.BatchNormalization()(x_1_loss_pattern)
            # x_1_delay = tf.keras.layers.BatchNormalization()(x_1_delay)

        x_1_loss_pattern = tf.keras.layers.LeakyReLU()(x_1_loss_pattern)
        # x_1_delay = tf.keras.layers.LeakyReLU() (x_1_delay)
        x_1_loss_pattern = tf.keras.layers.Flatten()(x_1_loss_pattern)
        # x_1_delay = tf.keras.layers.Flatten()(x_1_delay)

        x_1 = x_1_loss_pattern
        # x_1 = tf.keras.layers.concatenate([x_1_loss_pattern, x_1_delay])

        Dense1_1 = tf.keras.layers.Dense(config.Dense_out)
        x_1 = Dense1_1(x_1)
        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        #Dense2_1 = tf.keras.layers.Dense(self.action_dim, activation='softmax')
        Dense2_1 = tf.keras.layers.Dense(self.action_dim)
        logits = Dense2_1(x_1)
        if config.logit_clipping > 0:
            logits = config.logit_clipping*tf.keras.activations.tanh(logits)

        # Critic
        Conv1D_2_loss_pattern = tf.keras.layers.Conv1D(config.Conv1D_out, 1, strides=1, padding='valid', use_bias=not config.batch_norm)
        # Conv1D_2_delay = tf.keras.layers.Conv1D(config.Conv1D_out, 1, strides=1, padding='valid', use_bias=not config.batch_norm)
        x_2_loss_pattern = Conv1D_2_loss_pattern(inputs_loss_pattern)
        # x_2_delay = Conv1D_2_delay(inputs_delay)
        
        if config.batch_norm:
            x_2_loss_pattern = tf.keras.layers.BatchNormalization()(x_2_loss_pattern)
            # x_2_delay = tf.keras.layers.BatchNormalization()(x_2_delay)
        x_2_loss_pattern = tf.keras.layers.LeakyReLU()(x_2_loss_pattern)
        # x_2_delay = tf.keras.layers.LeakyReLU() (x_2_delay)
        x_2_loss_pattern = tf.keras.layers.Flatten()(x_2_loss_pattern)
        # x_2_delay = tf.keras.layers.Flatten()(x_2_delay)


        # x_2 = tf.keras.layers.concatenate([x_2_loss_pattern, x_2_delay])
        x_2 = x_2_loss_pattern
        Dense1_2 = tf.keras.layers.Dense(config.Dense_out)
        x_2 = Dense1_2(x_2)
        x_2 = tf.keras.layers.LeakyReLU()(x_2)
        if config.normalized_reward:
            Dense2_2 = tf.keras.layers.Dense(1, activation='sigmoid')
        else:
            Dense2_2 = tf.keras.layers.Dense(1)
        values = Dense2_2(x_2)

        self.model = tf.keras.models.Model(inputs, [logits, values])

        self.actor_model = tf.keras.models.Model(inputs, logits) # inputs: 2, 30, 5
        self.critic_model = tf.keras.models.Model(inputs, values)
    def create_policy_model(self, config):
        tf.keras.backend.set_floatx('float32')
        inputs = tf.keras.Input(shape=(self.input_dims[0], self.input_dims[1]))# not including the batch size
        inputs_loss_pattern = inputs
        # inputs_delay = inputs[:,1,:,:]
        # tf.reshape(inputs_loss_pattern, [1, self.input_dims[0], self.input_dims[1]])
        # tf.reshape(inputs_delay, [1, self.input_dims[0], self.input_dims[1]])
        
        Conv1D_1_loss_pattern = tf.keras.layers.Conv1D(config.Conv1D_out, 1, strides=1, padding='valid', use_bias=not config.batch_norm)
        # Conv1D_1_delay = tf.keras.layers.Conv1D(config.Conv1D_out, 1, strides=1, padding='valid', use_bias=not config.batch_norm)
        
        x_1_loss_pattern = Conv1D_1_loss_pattern(inputs_loss_pattern)
        # x_1_delay = Conv1D_1_delay(inputs_delay)

        if config.batch_norm:
            x_1_loss_pattern = tf.keras.layers.BatchNormalization()(x_1_loss_pattern)
            # x_1_delay = tf.keras.layers.BatchNormalization()(x_1_delay)

        x_1_loss_pattern = tf.keras.layers.LeakyReLU()(x_1_loss_pattern)
        # x_1_delay = tf.keras.layers.LeakyReLU() (x_1_delay)
        x_1_loss_pattern = tf.keras.layers.Flatten()(x_1_loss_pattern)
        # x_1_delay = tf.keras.layers.Flatten()(x_1_delay)

        x_1 = x_1_loss_pattern
        # x_1 = tf.keras.layers.concatenate([x_1_loss_pattern, x_1_delay])

        Dense1_1 = tf.keras.layers.Dense(config.Dense_out)
        x_1 = Dense1_1(x_1)
        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        Dense1_2 = tf.keras.layers.Dense(config.Dense_out)
        x_1 = Dense1_2(x_1)
        x_1 = tf.keras.layers.LeakyReLU()(x_1)
        #Dense2_1 = tf.keras.layers.Dense(self.action_dim, activation='softmax')
        Dense2_1 = tf.keras.layers.Dense(self.action_dim * (self.critical_num+1)) #output layer: 10*(K+1)
        logits = Dense2_1(x_1)
        if config.logit_clipping > 0:
            logits = config.logit_clipping*tf.keras.activations.tanh(logits)

        self.policy_model = tf.keras.models.Model(inputs, logits) 
    
    def value_loss_fn(self, rewards, values):
        advantages = tf.convert_to_tensor(rewards[:, None], dtype=tf.float32) - values
        value_loss = advantages ** 2
        value_loss = tf.reduce_mean(value_loss)
        
        return value_loss, advantages

    def policy_loss_fn(self, logits, actions, advantages, entropy_weight=0.01):
        advantages = tf.keras.backend.repeat_elements(advantages, rep=self.max_moves, axis=0)           #Duplicate advantages, shape = [batch_size*max_moves, 1]
        policy = tf.nn.softmax(logits)
        logits = tf.keras.backend.repeat_elements(logits, rep=self.max_moves, axis=0)                   #Duplicate logits and policy
        policy = tf.keras.backend.repeat_elements(policy, rep=self.max_moves, axis=0)                   #Make sure their shapes = [batch_size*max_moves, action_dim]
        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)                 #entropy already includes "-", shape = [batch_size*max_moves,]
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions,logits=logits)      #shape = [batch_size*max_moves,]  
        policy_loss = tf.expand_dims(policy_loss, -1) * tf.stop_gradient(advantages)                    #should be reshaped to [batch_size*max_moves, 1]
        policy_loss -= entropy_weight * tf.expand_dims(entropy, -1)
        policy_loss = tf.reduce_sum(policy_loss)
        
        return policy_loss, entropy

    def policy_loss_fn_with_log_epsilon(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-12):
        advantages = tf.keras.backend.repeat_elements(advantages, rep=self.max_moves, axis=0)               #Duplicate advantages, shape = [batch_size*max_moves, 1]
        policy = tf.nn.softmax(tf.multiply(logits, 1.0/math.sqrt(float(self.action_dim))))
        #policy = tf.nn.softmax(logits)
        policy = tf.keras.backend.repeat_elements(policy, rep=self.max_moves, axis=0)                       #Duplicate policy
        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]              #Make sure policy shape = [batch_size*max_moves, action_dim]
        entropy = -tf.reduce_sum(tf.multiply(policy, tf.math.log(tf.maximum(policy, log_epsilon))), 1, keepdims=True)       #shape=[batch_size*max_moves, 1]
        policy_loss = tf.math.log(tf.maximum(tf.reduce_sum(tf.multiply(policy, actions), 1, keepdims=True), log_epsilon))   #[batch_size*max_moves, 1]
        """ '-'advantages means gradient asdcend. same for entropy """
        policy_loss = tf.multiply(policy_loss, tf.stop_gradient(-advantages))                                               #[batch_size*max_moves, 1]
        policy_loss -= entropy_weight * entropy
        policy_loss = tf.reduce_sum(policy_loss)
        
        return policy_loss, entropy

    def policy_loss_fn_product_action(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-45):
        actions = tf.reshape(actions, [-1, self.max_moves, self.action_dim])                                    #[batch_size, max_moves, action_dim]
        policy = tf.nn.softmax(tf.multiply(logits, 1.0/math.sqrt(float(self.action_dim))))
        #policy = tf.nn.softmax(logits)
        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]                  #policy shape = [batch_size, action_dim]
        entropy = -tf.reduce_sum(tf.multiply(policy, tf.math.log(tf.maximum(policy, log_epsilon))), 1, keepdims=True)       #shape=[batch_size, 1]
        policy = tf.expand_dims(policy, -1)     #policy:[Batch_size, 6, 1], actions: [Batch_size, 1, 6]                                                               #[batch_size, action_dim, 1]
        policy_loss = tf.math.log(tf.maximum(tf.squeeze(tf.matmul(actions, policy)), log_epsilon))              #[batch_size, max_moves]
        # policy_loss = tf.reduce_sum(policy_loss, 1)                                              #[batch_size, 1]
        """ '-'advantages means gradient asdcend. same for entropy """
        policy_loss = tf.multiply(policy_loss, tf.stop_gradient(-advantages))                                   #[batch_size, 1]
        policy_loss -= entropy_weight * entropy
        policy_loss = tf.reduce_sum(policy_loss)
        
        return policy_loss, entropy

    def policy_loss_fn_multiactions(self, logits, actions, advantages, entropy_weight=0.01):
        # actions dimension: ( Batchsize, (K+1)*10 )
        #calculate policy
        policy = [] #shape ( 1, 10*(critical_num+1) )
        for i in range(self.critical_num + 1):
            policy.append(tf.nn.softmax(logits[:, self.action_dim*i:self.action_dim*(i+1)]))
        policy = tf.stack(policy, axis=1)# [Batchsize, 6, 10]
        policy = tf.reshape(policy, logits.shape)
        
        # logits = tf.keras.backend.repeat_elements(logits, rep=self.max_moves, axis=0)                   #Duplicate logits and policy
        # policy = tf.keras.backend.repeat_elements(policy, rep=self.max_moves, axis=0)                   #Make sure their shapes = [batch_size*max_moves, action_dim]
        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]

        #calculate cumulated entropy
        entropy = -tf.reduce_sum(tf.multiply(policy, tf.math.log(policy)), axis=1, keepdims=True)       #shape = [batch_size, 1]
        policy = tf.expand_dims(policy, -1)     #policy:[Batch_size, 60, 1], 
        actions = tf.expand_dims(actions, 1)    # actions: [Batch_size, 1, 60]                                                               #[batch_size, action_dim, 1]
        
        policy_loss = tf.squeeze(tf.matmul(actions, tf.math.log(policy))) #[Batch_size,]
        policy_loss = tf.expand_dims(policy_loss, -1)
        
        policy_loss = tf.multiply(policy_loss, tf.stop_gradient(-advantages))                                   #[batch_size, 1]
        policy_loss -= entropy_weight * entropy
        policy_loss = tf.reduce_sum(policy_loss)
        
        
        return policy_loss, entropy
    def policy_loss_fn_product_multiactions(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-45):
        # actions dimension: ( Batchsize, (K+1)*10 )
        #calculate policy
        policy = [] #shape ( 1, 10*(critical_num+1) )
        for i in range(self.critical_num + 1):
            policy.append(tf.nn.softmax(tf.multiply(logits[:, self.action_dim*i:self.action_dim*(i+1)], 1.0/math.sqrt(float(self.action_dim)))))
        policy = tf.stack(policy, axis=1)# [Batchsize, (critical_num+1), 10]
        policy = tf.reshape(policy, logits.shape)
        
        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]                  #policy shape = [batch_size, action_dim]
        
        entropy = -tf.reduce_sum(tf.multiply(policy, tf.math.log(tf.maximum(policy, log_epsilon))), 1, keepdims=True)       #shape=[batch_size, 1]
        policy = tf.expand_dims(policy, -1)     #policy:[Batch_size, 60, 1], actions: [Batch_size, 1, 60]  if K = 5                                                             #[batch_size, action_dim, 1]
        actions = tf.expand_dims(actions, 1)
        
        policy_loss = tf.squeeze(tf.matmul(actions, tf.math.log(tf.maximum(policy, log_epsilon)))) #[Batch_size,]                                            #[batch_size, 1]
        """ '-'advantages means gradient asdcend. same for entropy """
        policy_loss = tf.expand_dims(policy_loss, -1)
        assert advantages.shape == policy_loss.shape
        policy_loss = tf.multiply(policy_loss, tf.stop_gradient(-advantages))                                   #[batch_size, 1]
        policy_loss -= entropy_weight * entropy
        policy_loss = tf.reduce_sum(policy_loss)
        
        return policy_loss, entropy


    @tf.function
    def actor_critic_train(self, inputs, actions, rewards, entropy_weight=0.1):
        if self.attention_mask is not None:
            # Tracks the variables involved in computing the loss by using tf.GradientTape
            shape = transformer.get_shape_list(inputs, expected_rank=3)
            attention_masks = np.repeat(self.attention_mask, shape[0], axis=0)
            with tf.GradientTape() as tape:
                values = self.critic_model([inputs, attention_masks], training=True)
                value_loss, advantages = self.value_loss_fn(rewards, values)

            critic_gradients = tape.gradient(value_loss, self.critic_model.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_model.trainable_variables))

            with tf.GradientTape() as tape:
                logits = self.actor_model([inputs, attention_masks], training=True)
                #policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
                #policy_loss, entropy = self.policy_loss_fn_with_log_epsilon(logits, actions, advantages, entropy_weight)
                policy_loss, entropy = self.policy_loss_fn_product_action(logits, actions, advantages, entropy_weight)

            actor_gradients = tape.gradient(policy_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))
        else:
            # Tracks the variables involved in computing the loss by using tf.GradientTape
            with tf.GradientTape() as tape:
                values = self.critic_model(inputs, training=True)
                value_loss, advantages = self.value_loss_fn(rewards, values)

            critic_gradients = tape.gradient(value_loss, self.critic_model.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_model.trainable_variables))

            with tf.GradientTape() as tape:
                logits = self.actor_model(inputs, training=True)
                #policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
                #policy_loss, entropy = self.policy_loss_fn_with_log_epsilon(logits, actions, advantages, entropy_weight)
                policy_loss, entropy = self.policy_loss_fn_product_action(logits, actions, advantages, entropy_weight)

            actor_gradients = tape.gradient(policy_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))

        return value_loss, policy_loss, entropy, actor_gradients, critic_gradients
    
    @tf.function
    def policy_train(self, inputs, actions, advantages, entropy_weight=0.01):
        if self.attention_mask is not None:
            shape = transformer.get_shape_list(inputs, expected_rank=3)
            attention_masks = np.repeat(self.attention_mask, shape[0], axis=0)
            # Tracks the variables involved in computing the loss by using tf.GradientTape
            with tf.GradientTape() as tape:
                logits = self.model([inputs, attention_masks], training=True)
                #advantages = tf.convert_to_tensor(advantages[:, None], dtype=tf.float32)
                #policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
                #policy_loss, entropy = self.policy_loss_fn_with_log_epsilon(logits, actions, advantages, entropy_weight)
                policy_loss, entropy = self.policy_loss_fn_product_action(logits, actions, advantages, entropy_weight)
        else:
            # Tracks the variables involved in computing the loss by using tf.GradientTape
            with tf.GradientTape() as tape:
                logits = self.policy_model(inputs, training=True)                
                #advantages = tf.convert_to_tensor(advantages[:, None], dtype=tf.float32)
                #policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
                #policy_loss, entropy = self.policy_loss_fn_with_log_epsilon(logits, actions, advantages, entropy_weight)
                policy_loss, entropy = self.policy_loss_fn_product_multiactions(logits, actions, advantages, entropy_weight)
                # policy_loss, entropy = self.policy_loss_fn_multiactions(logits, actions, advantages, entropy_weight)

        gradients = tape.gradient(policy_loss, self.policy_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))

        return entropy, gradients, policy_loss

    @tf.function
    def actor_predict(self, inputs, policy_epsilon=1e-12):
        
        if self.attention_mask is not None:
            logits = self.actor_model([inputs, self.attention_mask], training=False)
        else:
            logits = self.actor_model(inputs, training=False)
        policy = tf.nn.softmax(logits)
        
        #policy = tf.maximum(policy, policy_epsilon)         # In case non zero elements of policy < config.max_moves

        return policy

    @tf.function
    def critic_predict(self, inputs):
        if self.attention_mask is not None:
            critic_outputs = self.critic_model([inputs, self.attention_mask], training=False)
        else:
            critic_outputs = self.critic_model(inputs, training=False)
        
        return critic_outputs

    @tf.function
    def policy_predict(self, inputs):
        if self.attention_mask is not None:
            logits = self.policy_model([inputs, self.attention_mask], training=False)
        else:
            logits = self.policy_model(inputs, training=False)
        policy = [] #shape ( 1, (10*critical_num+1) )
        for i in range(self.critical_num + 1):
            policy.append(tf.nn.softmax(logits[:, self.action_dim*i:self.action_dim*(i+1)]))
        policy = tf.stack(policy, axis=1)
        policy = tf.reshape(policy, logits.shape)
        return policy

    def preprocess(self, _s_batch, _r_batch):
        #Instead of computing gradients for each action,
        #batch all actions and compute one set of gradients.
        batch_size = len(_s_batch)
        assert batch_size == len(_r_batch), ('batch size does not match', len(_s_batch), len(_r_batch))

        s_batch = []
        r_batch = []
        for i in range(batch_size):
            s_batch += [_s_batch[i] for _ in range(self.max_moves)]
            r_batch += [_r_batch[i] for _ in range(self.max_moves)]
        
        return s_batch, r_batch
 
    def restore_ckpt(self, checkpoint=''):
        if checkpoint == '':
            checkpoint = self.manager.latest_checkpoint
        else:
            checkpoint = self.ckpt_dir+'/'+checkpoint

        self.ckpt.restore(checkpoint).expect_partial()
        if checkpoint:
            step = int(self.ckpt.step)
            print("Restored from {}".format(checkpoint), step)
        else:
            step = 0
            print("Initializing from scratch.")

        return step

    def save_ckpt(self, _print=False):
        save_path = self.manager.save()
        if _print:
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def inject_summaries(self, summary_dict, step):
        with self.writer.as_default():
            for summary in summary_dict:
                tf.summary.scalar(summary, summary_dict[summary], step=step)
            self.writer.flush()

    def save_hyperparams(self, config):
        fp = self.ckpt_dir + '/hyper_parameters'

        hparams = {k:v for k, v in inspect.getmembers(config)
            if not k.startswith('__') and not callable(k)}

        if os.path.exists(fp):
            f = open(fp, 'r')
            match = True
            for line in f:
                idx = line.find('=')
                if idx == -1:
                    continue
                k = line[:idx-1]
                v = line[idx+2:-1]
                if v != str(hparams[k]):
                    match = False
                    print(k, v, hparams[k])
                    break
            f.close()
            if match:
                return
            
            f = open(fp, 'a')
        else:
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            f = open(fp, 'w+')

        for k, v in hparams.items():
            f.writelines(k + ' = ' + str(v) + '\n')
        f.writelines('\n')
        print("Save hyper parameters: %s" % fp)
        f.close()