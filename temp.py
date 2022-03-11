import numpy as np
# from MOS import mos_score
# from RS_decode import decode
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


from get_vq import get_video_quality
tm_history = 5
state_length = 90
tm_cnt = 500
input_state_dims = [state_length, tm_history]
train_tm_cnt = tm_cnt - tm_history + 1
trace_file = './data/video_data.txt'

def load_traffic():
    f_trace = open(trace_file, 'r')
    loss_pattern = []
    for line in f_trace:
        pattern = [float(v) for v in line.strip().split(' ')]
        line_length = len(pattern)
        assert line_length == state_length #config.state_length
        loss_pattern.append(pattern)
    f_trace.close()
    loss_pattern = np.array(loss_pattern)
    pattern_shape = loss_pattern.shape #pattern_cnt * config.state_length
    pattern_cnt = pattern_shape[0]
    input_pattern_dims = [pattern_shape[1], tm_history]
    print('matrices: %d, traffic dims: [%d, %d]\n'%(pattern_cnt, input_pattern_dims[0], input_pattern_dims[1]))
    # print(loss_pattern)

    
    return loss_pattern

def generate_loss_delay(loss_pattern):
    state_loss = np.zeros(([train_tm_cnt]+ [1] + input_state_dims), dtype=np.float32) #[N, 1, state_length, tm_history]
    
    for i in range(train_tm_cnt):
        tm_idx = i+tm_history-1
        for h in range(tm_history):
            state_loss[i,:,:,h] = loss_pattern[tm_idx-h]
            
    
    state_data = state_loss
    print(state_data.shape)
    return state_loss


loss_pattern= load_traffic()
state_loss = generate_loss_delay(loss_pattern)
loss_pattern_current = state_loss[50,:,:,0]

vmaf = get_video_quality(loss_pattern_current)
print(vmaf)
# print(decode(loss_pattern_current.squeeze(), network_delay_current.squeeze(), 4, 2))