import tensorflow as tf
input_dims = [30, 5]
action_dim = 8
batch_norm = False
Conv1D_out = 32
Dense_out = 128
normalized_reward = False
def create_actor_critic_model():
    tf.keras.backend.set_floatx('float32')
    inputs = tf.keras.Input(shape=(1, input_dims[0], input_dims[1]))
    print(inputs.shape)
    inputs_loss_pattern = inputs[0]
    # inputs_delay = inputs[1]
    # Actor
    Conv1D_1_loss_pattern = tf.keras.layers.Conv1D(32, 3, strides=3, padding='valid', use_bias=not batch_norm)
    # Conv1D_1_delay = tf.keras.layers.Conv1D(32, 3, strides=3, padding='valid', use_bias=not batch_norm)
    
    x_1_loss_pattern = Conv1D_1_loss_pattern(inputs_loss_pattern)
    # x_1_delay = Conv1D_1_delay(inputs_delay)

    if batch_norm:
        x_1_loss_pattern = tf.keras.layers.BatchNormalization()(x_1_loss_pattern)
        # x_1_delay = tf.keras.layers.BatchNormalization()(x_1_delay)

    x_1_loss_pattern = tf.keras.layers.LeakyReLU()(x_1_loss_pattern)
    # x_1_delay = tf.keras.layers.LeakyReLU() (x_1_delay)
    x_1_loss_pattern = tf.keras.layers.Flatten()(x_1_loss_pattern)
    # x_1_delay = tf.keras.layers.Flatten()(x_1_delay)

    
    
    # x_1 = tf.keras.layers.concatenate([x_1_loss_pattern, x_1_delay])
    x_1 = x_1_loss_pattern
    Dense1_1 = tf.keras.layers.Dense(Dense_out)
    x_1 = Dense1_1(x_1)
    x_1 = tf.keras.layers.LeakyReLU() (x_1)
    #Dense2_1 = tf.keras.layers.Dense(self.action_dim, activation='softmax')
    Dense2_1 = tf.keras.layers.Dense(action_dim)
    logits = Dense2_1(x_1)

    # Critic
    Conv1D_2_loss_pattern = tf.keras.layers.Conv1D(Conv1D_out, 3, strides=3, padding='valid', use_bias=not batch_norm)
    # Conv1D_2_delay = tf.keras.layers.Conv1D(Conv1D_out, 3, strides=3, padding='valid', use_bias=not batch_norm)
    x_2_loss_pattern = Conv1D_2_loss_pattern(inputs_loss_pattern)
    # x_2_delay = Conv1D_2_delay(inputs_delay)
    
    if batch_norm:
        x_2_loss_pattern = tf.keras.layers.BatchNormalization()(x_2_loss_pattern)
        # x_2_delay = tf.keras.layers.BatchNormalization()(x_2_delay)
    x_2_loss_pattern = tf.keras.layers.LeakyReLU()(x_2_loss_pattern)
    # x_2_delay = tf.keras.layers.LeakyReLU() (x_2_delay)
    x_2_loss_pattern = tf.keras.layers.Flatten()(x_2_loss_pattern)
    # x_2_delay = tf.keras.layers.Flatten()(x_2_delay)

    # x_2 = tf.keras.layers.concatenate([x_2_loss_pattern, x_2_delay])
    x_2 = x_2_loss_pattern
    Dense1_2 = tf.keras.layers.Dense(Dense_out)
    x_2 = Dense1_2(x_2)
    x_2 = tf.keras.layers.LeakyReLU()(x_2)
    if normalized_reward:
        Dense2_2 = tf.keras.layers.Dense(1, activation='sigmoid')
    else:
        Dense2_2 = tf.keras.layers.Dense(1)
    values = Dense2_2(x_2)

    model = tf.keras.models.Model(inputs, [logits, values])

    actor_model = tf.keras.models.Model(inputs, logits)
    critic_model = tf.keras.models.Model(inputs, values)
    return model


model = create_actor_critic_model()
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])