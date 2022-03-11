class NetworkConfig(object):
  scale = 100

  max_step = 1200
  
  initial_learning_rate = 0.001
  learning_rate_minimum = 0.0001
  learning_rate_decay_rate = 0.96
  learning_rate_decay_step = 150
  moving_average_decay = 0.9999
  entropy_weight = 0.1

  save_step = 10
  max_to_keep = 1000

  #Transformer
  embedding_dim = None
  num_layers = 3
  num_attention_heads = 8
  intermediate_dim = 256
  #transformer_normalization = 'LayerNormalization'
  transformer_normalization = 'BatchNormalization'
  #transformer_normalization = None

  #Conv
  Conv1D_out = 8
  Dense_out = 256
  batch_norm = False
  
  optimizer = 'RMSprop'
  #optimizer = 'Adam'

  logit_clipping = 10           #10 or 0, = 0 means logit clipping is disable

  l2_regularizer = 0.001

class Config(NetworkConfig):
  version = 'RL-RS_1.0'

  

  # model = 'actor_critic'
  model = 'pure_policy' # only use this one.
  
  graph_extractor = 'Conv'


  trace_file = 'video_RL_train.txt'
  # test_trace_file = 'video_RL_testing.txt'
  test_trace_file = 'test_t.txt'
  # test_trace_file = 'Emulation_video_test/mother_daughter/mother_daughter_0.15.txt'

  
  state_length = 90 # the length of loss pattern in one state.
  avg_matrices_num = 1
  tm_history = 5
  predict_interval = 2

  num_agents = 5

  max_moves = 1
  critical_num = 15
  action_dim = 10
  actions_pool = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


  normalized_reward = False

  softmax_temperature = False

  priority_sampling = True

  # For pure policy
  baseline = 'avg'          #avg, best

  num_iter = 900


def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
