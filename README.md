# RL-AFEC: Adaptive Forward Error Correction for Real-time Video Communication Based on Reinforcement Learning
This is a Tensorflow implementation of RL-AFEC.
# Prerequisites
Python 3.6.8, Tensorflow v2.2.0 (CPU version)

# Training
To train an RL-AFEC agent, put the trace file (e.g., video_RL_train.txt) in data/, then specify the file name in config.py, i.e., trace_file = 'video_RL_train.txt' 

In game.py, let function reward() return r:
```python
def reward(self, tm_idx, actions):    
        ...
        return r
        #return vmaf, bw_waste, loss_pattern_after #for testing
```
then run
```python
python3 train.py
```
# Testing
To test the trained policy on a set of test traces, put the test trace file (e.g., video_RL_testing.txt) in data/, then specify the file name in config.py, i.e., test_trace_file = 'video_RL_testing.txt'

In game.py, let function reward() return vmaf, bw_waste, loss_pattern_after:
```python
def reward(self, tm_idx, actions):    
        ...
        #return r
        return vmaf, bw_waste, loss_pattern_after #for testing
```
then run
```python
python3 test.py
```
