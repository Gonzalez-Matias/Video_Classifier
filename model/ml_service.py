import json
import time
import numpy as np
import redis
import settings
from keras.models import load_model
from utils import get_frames
import os
import glob

db = redis.Redis(
                host =settings.REDIS_IP,
                port =settings.REDIS_PORT,
                db = settings.REDIS_DB_ID,
                )
model = load_model("weights/model.07-1.91.h5")
mv_classes = sorted(["Static","Motion","Pull","Push"])
sc_classes = sorted(["MS","CS","ECS","FS","LS"])

def predict(new_name):
    """
    Load video from the corresponding path based on the video name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    video_name : str
        Video filename.

    Returns
    -------
    prediction : dict
        Model predicted class as a string
    """
    pattern = settings.UPLOAD_FOLDER+"*:{name}"
    scenes_dir = sorted(glob.glob(pattern.format(name=new_name)))
    prediction = {}

    for scene in scenes_dir:
        scene_name = os.path.split(scene)[-1].replace(".mp4","").split(":")[-2]
        vid = get_frames(scene)
        vid = np.expand_dims(vid, axis=0)
        sc_pred, mv_pred = model.predict(vid)
        sc_pred = sc_classes[np.array(sc_pred).argmax()]
        mv_pred = mv_classes[np.array(mv_pred).argmax()]
        prediction[scene_name] = {"sc_pred": sc_pred, "mv_pred": mv_pred}

    return prediction


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load video from the corresponding folder based on the video name
    received, then, run our ML model to get predictions.
    """
    while True:

        _, new_job = db.brpop(settings.REDIS_QUEUE)
        new_job = json.loads(new_job)

        prediction = predict(new_job["new_name"])
        
        response = json.dumps(prediction)
        db.set(new_job["id"], response)

        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    print("Launching ML service...")
    classify_process()
