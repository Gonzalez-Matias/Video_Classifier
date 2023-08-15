"""
VideoFrameGenerator - Simple Generator
--------------------------------------
A simple frame generator that takes distributed frames from
videos. It is useful for videos that are scaled from frame 0 to end
and that have no noise frames.
"""

import glob #<-----Remove this
import logging
import os
from math import floor
from typing import Iterable, Optional
from cv2 import VideoCapture
from cv2 import resize
from cv2 import CAP_PROP_FRAME_COUNT
from random import sample
from numpy import array
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array)
from tensorflow.keras.utils import Sequence

log = logging.getLogger()


class VideoFrameGenerator(Sequence):  # pylint: disable=too-many-instance-attributes
    """
    Create a generator that return batches of frames from video
    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes_sc: list of str, classes of scale to infer
    - classes_mv: list of str, classes of movement to infer
    - get_item: str, depending of what you label you want as a response \
        options:"sc", "mv", "both" it gives both from default
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - pattern: string, directory path with '{classname}' inside that \
        will be replaced by one of the class list
    - use_header: bool, default to True to use video header to read the \
        frame count if possible
    - seed: int, default to None, keep the seed value for split
    You may use the "classes" property to retrieve the class list afterward.
    The generator has that properties initialized:
    - classes_count: number of classes that the generator manages
    - files_count: number of video that the generator can provides
    - classes: the given class list
    - files: the full file list that the generator will use, this \
        is usefull if you want to remove some files that should not be \
        used by the generator.
    """

    def __init__(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches,too-many-arguments
        self,
        rescale: float = 1 / 255.0,
        nb_frames: int = 5,
        classes_sc: list = None,
        classes_mv: list = None,
        get_item: str = "both",
        batch_size: int = 32,
        use_frame_cache: bool = False,
        target_shape: tuple = (224, 224),
        shuffle: bool = True,
        transformation: Optional[ImageDataGenerator] = None,
        split_test: float = None,
        split_val: float = None,
        nb_channel: int = 3,
        pattern: str = "data/Videos/train/*_{class_sc}:{class_mv}.mp4",
        use_headers: bool = True,
        seed=None,
        **kwargs,
    ):

        self.pattern = pattern

        # should be only RGB or Grayscale
        assert nb_channel in (1, 3)

        if (classes_sc is None) | (classes_mv is None):
            classes_sc, classes_mv = self._discover_classes()

        # we should have classes
        if (len(classes_sc) == 0) | (len(classes_mv) == 0):
            log.warn(
                "You didn't provide classes list or "
                "we were not able to discover them from "
                "your pattern.\n"
                "Please check if the path is OK."
            )

        # shape size should be 2
        assert len(target_shape) == 2

        # split factor should be a propoer value
        if split_val is not None:
            assert 0.0 < split_val < 1.0

        if split_test is not None:
            assert 0.0 < split_test < 1.0

        self.use_video_header = use_headers

        # then we don't need None anymore
        split_val = split_val if split_val is not None else 0.0
        split_test = split_test if split_test is not None else 0.0

        # be sure that classes are well ordered
        classes_sc.sort()
        classes_mv.sort()

        self.rescale = rescale
        self.classes_sc = classes_sc
        self.classes_mv = classes_mv
        self.get_item = get_item
        self.batch_size = batch_size
        self.nbframe = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation
        self.use_frame_cache = use_frame_cache

        self._random_trans = []
        self.__frame_cache = {}
        self.files = []
        self.validation = []
        self.test = []

        np.random.seed(seed)
        for cls_sc in classes_sc:
            for cls_mv in classes_mv:
                for dir in os.listdir(f"{os.path.split(pattern)[:-1][0]}"):
                    pat = os.path.join(os.path.split(pattern)[:-1][0],dir,os.path.split(pattern)[-1])
                    self.files += glob.glob(pat.format(class_sc=cls_sc, class_mv=cls_mv))

        # build indexes
        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)
        self.classes_sc_count = len(classes_sc)
        self.classes_mv_count = len(classes_mv)

        # to initialize transformations and shuffle indices
        if "no_epoch_at_init" not in kwargs:
            self.on_epoch_end()

        self._current = 0
        self._framecounters = {}

    def _discover_classes(self):
        path = os.path.split(self.pattern)[:-1][0]
        video_paths = []
        for dirpath, _, files in os.walk(path):
            for x in files:
                if x.endswith(".mp4"):
                    video_paths.append(os.path.join(dirpath, x))
        labels_sc = [shot.replace(".mp4","").split(":")[0].split("_")[-1] for shot in video_paths]
        labels_mv = [shot.replace(".mp4","").split(":")[-1] for shot in video_paths]

        classes_sc = list(set(labels_sc))
        classes_mv = list(set(labels_mv))

        return classes_sc, classes_mv

    def next(self):
        """ Return next element"""
        elem = self[self._current]
        self._current += 1
        if self._current == len(self):
            self._current = 0
            self.on_epoch_end()

        return elem

    def on_epoch_end(self):
        """ Called by Keras after each epoch """

        if self.transformation is not None:
            self._random_trans = []
            for _ in range(self.files_count):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return int(np.floor(self.files_count / self.batch_size))


    def __getitem__(self, index):
        classes_sc = self.classes_sc
        classes_mv = self.classes_mv

        labels_sc = []
        labels_mv = []
        images = []
        images_sal = []

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        for i in indexes:

            video = self.files[i]
            classname_sc, classname_mv = self._get_classname(video)

            # create a label array and set 1 to the right column
            label_sc = np.zeros(len(classes_sc))
            label_mv = np.zeros(len(classes_mv))
            col_sc = classes_sc.index(classname_sc)
            col_mv = classes_mv.index(classname_mv)
            label_sc[col_sc] = 1.0
            label_mv[col_mv] = 1.0

            if video not in self.__frame_cache:
                frames = self._get_frames(video)
                if frames is None:
                    # avoid failure, nevermind that video...
                    continue

                # add to cache
                if self.use_frame_cache:
                    self.__frame_cache[video] = frames


            else:
                frames = self.__frame_cache[video]

            # add the sequence in batch
            images.append(frames)
            labels_sc.append(label_sc)
            labels_mv.append(label_mv)
        labels_sc = np.array(labels_sc)
        labels_mv = np.array(labels_mv)
        list_lab_sc = []
        list_lab_mv = []
        for l in range(0,len(labels_sc)):
            label_sc = np.zeros(5)
            label_sc[labels_sc[l].argmax()] = 1
            label_mv = np.zeros(4)
            label_mv[labels_mv[l].argmax()] = 1
            list_lab_sc.append(label_sc)
            list_lab_mv.append(label_mv)
        if self.get_item == "sc":
            frame_n = sample(range(self.nbframe), 1)
            return np.array(images)[:,frame_n,:,:,:][:,0,:,:,:], np.array(list_lab_sc)

        if self.get_item == "mv":
            frame_n = sample(range(self.nbframe), 1)
            return np.array(images), np.array(list_lab_mv)

        return {"Input":np.array(images)}, {'sc_output': np.array(list_lab_sc), 'mv_output': np.array(list_lab_mv)}

    def _get_classname(self, video: str) -> str:
        """ Find classname from video filename following the pattern """

        # work with real path
        video = os.path.realpath(video)
        classname_sc = video.replace(".mp4","").split(":")[0].split("_")[-1]
        classname_mv = video.replace(".mp4","").split(":")[-1]

        return [classname_sc, classname_mv]


    def _get_frames(
        self, video
    ) -> Optional[Iterable]:

        cap = cv.VideoCapture(video)
        total_frames = int(cap.get(CAP_PROP_FRAME_COUNT))

        if total_frames >= self.nbframe:
            usedF = sorted(sample(range(total_frames), self.nbframe))
            frames = [resize(cap.read()[1], self.target_shape) for i in range(max(usedF)+1) if i in usedF]
        else:
            return None

        cap.release()
        return np.array(frames)

