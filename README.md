# RL-AFEC: Adaptive Forward Error Correction for Real-time Video Communication Based on Reinforcement Learning
This is a Tensorflow implementation of RL-AFEC
# Prerequisites
Python 3.8.5, Tensorflow v2.2.0 (CPU version)

# Training
To train an RL-AFEC agent, put the trace file (e.g., video_RL_train.txt) in data/, then specify the file name in config.py, i.e., trace_file = 'video_RL_train.txt' and then run
```python
python3 train.py

# Testing
To test the trained policy on a set of test traces, put the test trace file (e.g., video_RL_testing.txt) in data/, then specify the file name in config.py, i.e., test_traffic_file = 'video_RL_testing.txt', and then run
```python
python3 test.py
