from hashlib import md5
from pytube import YouTube
import settings

def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files.

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    if filename == None:
        return False

    exten = filename.split(".")[-1]

    return exten.lower() in {"mp4"}


def get_file_hash(file):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """

    hashed_file = md5(file.stream.read()).hexdigest()
    file.stream.seek(0)

    return f"{hashed_file}.mp4"


def download_from_youtube(url):
    SAVE_PATH = settings.UPLOAD_FOLDER
    try:
        yt = YouTube(url)
        d_video = yt.streams.get_by_resolution(resolution="360p")
        if not d_video:
            mp4files = yt.streams.filter(file_extension='mp4')
            low_res = int(mp4files.get_lowest_resolution().resolution.replace("p",""))
            high_res = int(mp4files.get_highest_resolution().resolution.replace("p",""))
            if 360 < low_res:
                res = low_res
                d_video = yt.streams.get_by_resolution(resolution=res)
            else:
                res = high_res
                d_video = yt.streams.get_by_resolution(resolution=res)
        save_path = d_video.download(SAVE_PATH)
        
        return save_path
    except:
        return str(f"LinkError: {url}")