import time
import settings
import json
import redis
from uuid import uuid4

db = redis.Redis(
                 host =settings.REDIS_IP,
                 port =settings.REDIS_PORT,
                 db = settings.REDIS_DB_ID
                )


def model_predict(new_name):
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    prediction: {scene_001:{sc_pred: str, mv_pred: str},
                 scene_002:{sc_pred: str, mv_pred: str}
                }
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    job_id = str(uuid4())

    job_data = {
                "id":job_id,
                "new_name":new_name
               }

    db.rpush(settings.REDIS_QUEUE,json.dumps(job_data))

    output=None
    while output==None:
        output = db.get(job_data["id"])

        time.sleep(settings.API_SLEEP)

    
    output = json.loads(output)
    prediction = output

    db.delete(job_data["id"],settings.REDIS_QUEUE)
    output=None
    
    return prediction
