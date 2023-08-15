import cv2
from cv2 import CAP_PROP_FRAME_COUNT
from cv2 import resize
import numpy as np

def get_frames(video):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    if total_frames >= 8:
        usedF = [int(round(num,0)) for num in np.linspace(0,total_frames,8,endpoint=False)]
        frames = [resize(cap.read()[1], (224,224)) for i in range(max(usedF)+1) if i in usedF]
    else:
        return None
    cap.release()
    return np.array(frames)