import numpy as np
from video_quality import load_model


def get_video_quality(loss_pattern):

    NN_model = load_model()
    vmaf = NN_model.predict(loss_pattern)
    if vmaf < 0:
        vmaf = np.array([[0]])
    elif vmaf > 100:
        vmaf = np.array([[100]])
    
    return vmaf

