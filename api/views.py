import os
import shutil
import cv2
import utils
import io
from middleware import model_predict
from utils import download_from_youtube
from werkzeug.datastructures import FileStorage
from os import path
import settings
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import json
from flask import (
    Blueprint,
    flash,
    send_from_directory,
    make_response,
    redirect,
    render_template,
    request,
    jsonify
)



router = Blueprint("app_router", __name__, template_folder="templates")


@router.route("/", methods=["GET"])
def index():
    """
    Index endpoint, renders our HTML code.
    """
    return render_template("newindex.html")


@router.route("/", methods=["POST"])
def upload_video():
    """
    Function used in our frontend so we can upload and show a video.
    When it receives a video from the UI, it also calls our ML model to
    get and display the predictions.
    """

    if "url" not in request.form:
        flash("Please submit a correct url")
        return redirect(request.url)

    try:
        file = request.files["file"]
    except:
        file = FileStorage(None)
    url = request.form['url']
    if (file.filename == "") and (url == None):
        flash("No video selected for uploading")
        return redirect(request.url)

    if (file and utils.allowed_file(file.filename)) or (url != None):

        if url:
            save_path = download_from_youtube(url)
            with open(save_path, "rb") as vid:
                a = vid.read()
                BytesIO = io.BytesIO(a)
                file = FileStorage(BytesIO)

        new_name = utils.get_file_hash(file)

        file.save(path.join(settings.UPLOAD_FOLDER, new_name))

        video_path = path.join(settings.UPLOAD_FOLDER, new_name)
        scene_list = detect(video_path, ContentDetector())
        if len(scene_list) > 0:
            split_video_ffmpeg(video_path, scene_list, output_file_template =f'static/uploads/Scene-$SCENE_NUMBER:{new_name}')
        else:
            new_path = os.path.join(os.path.split(video_path)[-2],str("Scene-001:"+os.path.basename(video_path)))
            time = cv2
            shutil.copy(video_path, new_path)

        prediction = model_predict(new_name)
        
        i=0
        scenes = list(prediction.keys())
        cam = cv2.VideoCapture(video_path)
        time = round(cam.get(cv2.CAP_PROP_FRAME_COUNT)/cam.get(cv2.CAP_PROP_FPS),2)
        if len(scene_list) > 0:
            for scene in scene_list:
                prediction[scenes[i]]["start"]=round(scene[0].get_seconds(),2)
                prediction[scenes[i]]["finish"]=round(scene[1].get_seconds(),2)
                i += 1
        else:
            prediction[scenes[i]]["start"]=0
            prediction[scenes[i]]["finish"]=time

        prediction = json.dumps(prediction)

        return render_template('newindex.html', filename=url.split("=")[1] , prediction=prediction)

    else:
        flash("Allowed video types are -> mp4")
        return redirect(request.url)


@router.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(settings.UPLOAD_FOLDER, filename)

                               

@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    file : str
        Input url from youtube video that we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "prediction": str,
                "score": float,
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "prediction" model predicted class as string.
        - "score" model confidence score for the predicted class as float.
    """
    rpse = {'success': False}

    bad_request = make_response(jsonify(rpse), 400)

    if "file" not in request.files:
        return bad_request

    file = request.files["file"]
    url = request.form['url']
    
    if (file.filename == "") and (url == None):
        return redirect(request.url)


    if (file and utils.allowed_file(file.filename)) or (url != None):

        if url:
            save_path = download_from_youtube(url)
            with open(save_path, "rb") as vid:
                a = vid.read()
                BytesIO = io.BytesIO(a)
                file = FileStorage(BytesIO)

        new_name = utils.get_file_hash(file)

        if not path.exists(path.join(settings.UPLOAD_FOLDER, new_name)):
            file.save(path.join(settings.UPLOAD_FOLDER, new_name))

        video_path = path.join(settings.UPLOAD_FOLDER, new_name)
        scene_list = detect(video_path, ContentDetector())

        if len(scene_list) > 0:
            split_video_ffmpeg(video_path, scene_list, output_file_template =f'static/uploads/Scene-$SCENE_NUMBER:{new_name}')
        else:
            new_path = os.path.join(os.path.split(video_path)[-2],str("Scene-001:"+os.path.basename(video_path)))
            time = cv2
            shutil.copy(video_path, new_path)

        prediction = model_predict(new_name)
        
        i=0
        scenes = list(prediction.keys())
        cam = cv2.VideoCapture(video_path)
        time = round(cam.get(cv2.CAP_PROP_FRAME_COUNT)/cam.get(cv2.CAP_PROP_FPS),2)
        if len(scene_list) > 0:
            for scene in scene_list:
                prediction[scenes[i]]["start"]=round(scene[0].get_seconds(),2)
                prediction[scenes[i]]["finish"]=round(scene[1].get_seconds(),2)
                i += 1
        else:
            prediction[scenes[i]]["start"]=0
            prediction[scenes[i]]["finish"]=time


        rpse = prediction
        resp = make_response(jsonify(rpse), 200)

        return resp

    else:
        return bad_request