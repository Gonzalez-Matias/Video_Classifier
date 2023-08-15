import os
import argparse
import cv2
from cv2 import VideoCapture


def parse_args():
    """
    Use argparse to get path to the json and if you want labeled names.
    """
    ap = argparse.ArgumentParser(
        description="Use argparse to get path to the json and if you want labeled names."
        )
    ap.add_argument( "-n", "--split_json", required=True,
        type=str,
        help="Path to json file with the split distribution.",
    )
    ap.add_argument(
        "-l", "--labeled", required=True,
        type=str,
        help="Set True if you want labeled names or False if you don't",
    )
    args = vars(ap.parse_args())
    return args


def make_labeled_path(
    dict: dict,
    part: str,
    trailer: str,
    shot: str,
):
    label_scale = dict[part][trailer][shot]["scale"]["label"]
    movement_scale = dict[part][trailer][shot]["movement"]["label"]
    file_dst_labeled_path = (f"data/Videos/{part}/{trailer}/shot_{shot}_{label_scale}:{movement_scale}.mp4")
    return file_dst_labeled_path

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dst_path(
    dict: dict = None,
    labeled: str = "False",
    part: str = None,
    trailer: str = None,
    shot: str = None,
):
    if labeled == "True":
        file_dst_path = make_labeled_path(dict, part, trailer, shot)
    else:
        file_dst_path = (f"data/Videos/{part}/{trailer}/shot_{shot}.mp4")
    return file_dst_path    

def link_shot(
    dict: dict,
    labeled: str,
    part: str,
    trailer: str,
    shot: str,  
):
    file_src_path = (f"/home/data/Videos/{trailer}/shot_{shot}.mp4")
    if int(cv2.VideoCapture(file_src_path).get(cv2.CAP_PROP_FRAME_COUNT)) >= 8:
        file_dst_path = make_dst_path(dict, labeled, part, trailer, shot)
        create_path(f"data/Videos/{part}/{trailer}")
        os.link(file_src_path,file_dst_path)
    else:
        shot_l = os.path.basename(file_src_path)
        trailer_l = os.path.basename(os.path.split(file_src_path)[0])
        print(f"The {shot_l} clip from {trailer_l} trailer was omitted because it didn't have enough frames")

def train_val_test_split(json_path, labeled):
    with open(json_path) as json_split:
        split = eval(json_split.read())
    for part in split.keys():
        if part == "extras":
            for trailer in split[part].keys():
                shot_list = split[part][trailer]["list"]
                for shot in shot_list:
                    trailer_name = trailer.replace("_aug","")
                    file_src_path = (f"/home/data/extra/{trailer_name}/{shot}.mp4")
                    file_dst_path = (f"data/Videos/train/{trailer}/{shot}.mp4")
                    if int(cv2.VideoCapture(file_src_path).get(cv2.CAP_PROP_FRAME_COUNT)) >= 8:
                        create_path(f"data/Videos/train/{trailer}")
                        os.link(file_src_path,file_dst_path)
                    else:
                        print(f"The {shot} clip from {trailer} trailer was omitted because it didn't have enough frames")
                        continue
        elif part == "extras_val":
            for trailer in split[part].keys():
                shot_list = split[part][trailer]["list"]
                for shot in shot_list:
                    trailer_name = trailer.replace("_aug","")
                    file_src_path = (f"/home/data/extra_val/{trailer_name}/{shot}.mp4")
                    file_dst_path = (f"data/Videos/val/{trailer}/{shot}.mp4")
                    if int(cv2.VideoCapture(file_src_path).get(cv2.CAP_PROP_FRAME_COUNT)) >= 8:
                        create_path(f"data/Videos/val/{trailer}")
                        os.link(file_src_path,file_dst_path)
                    else:
                        print(f"The {shot} clip from {trailer} trailer was omitted because it didn't have enough frames")
                        continue
        else:        
            for trailer in split[part].keys():
                for shot in split[part][trailer].keys():
                    link_shot(split, labeled, part, trailer, shot)
    return print("Finish splitting")


if __name__ == "__main__":
    args = parse_args()
    split_json, labeled = args['split_json'], args['labeled']
    train_val_test_split(split_json, labeled)
