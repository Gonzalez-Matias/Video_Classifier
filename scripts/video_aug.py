import argparse
import moviepy.editor as mpy
import os

def parse_args():
    """
    Use argparse to get path to the json and if you want labeled names.
    """
    ap = argparse.ArgumentParser(
        description="Use argparse to get path to the json and if you want labeled names."
        )
    ap.add_argument( "-n", "--split_json", required=True,
        type=str,
        help="Path to json file with the split distribution.",)
    args = vars(ap.parse_args())
    return args

def add_videos(path, scale_type, trailer):
  name= os.path.basename(path).replace(".mp4","")
  video = mpy.VideoFileClip(path)
  slide_pull = video.resize(lambda t: (1+(0.65)) - (0.65/video.duration) * t).set_position(('center', 'center'))
  slide_push = video.resize(lambda t: 1 + (0.65/video.duration) * t).set_position(('center', 'center'))
  result_pull = mpy.CompositeVideoClip([slide_pull], size=tuple(video.size))
  result_push = mpy.CompositeVideoClip([slide_push], size=tuple(video.size))
  result_pull.write_videofile(f"/home/data/extra_2/{trailer}/{name}_{scale_type}:Pull.mp4",fps=video.fps)
  result_push.write_videofile(f"/home/data/extra_2/{trailer}/{name}_{scale_type}:Push.mp4",fps=video.fps)
  return

def aug_videos(json_path):
    with open(json_path) as json_split:
        split = eval(json_split.read())
    for trailer in split["train"].keys():
        for shot in split["train"][trailer].keys():
            if split["train"][trailer][shot]["movement"]["label"] == "Static":
                new_path = f"/home/data/extra_2/{trailer}"
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                scale = split["train"][trailer][shot]["scale"]["label"]
                add_videos(path=f"/home/data/Videos/{trailer}/shot_{shot}.mp4", scale_type=scale, trailer=trailer)


if __name__ == "__main__":
    args = parse_args()
    split_json = args['split_json']
    aug_videos(split_json)