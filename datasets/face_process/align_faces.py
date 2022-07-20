#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
from imutils.face_utils import FaceAligner
import dlib
import cv2
import os
import pathlib


class FaceData():
    """Ref:
        * https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
        * https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c
    """

    def __init__(self, out_size=256):
        # self.detector = dlib.get_frontal_face_detector()
        self.detector = dlib.cnn_face_detection_model_v1(os.path.join(pathlib.Path(__file__).parent.absolute(), 'mmod_human_face_detector.dat'))
        predictor = dlib.shape_predictor(os.path.join(pathlib.Path(__file__).parent.absolute(), 'shape_predictor_68_face_landmarks.dat'))
        self.fa = FaceAligner(predictor, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=out_size)

    def normalize(self, img_path):
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = self.detector(gray, 1)  # 1 is the number of times it should upsample the image.
        faceAligned = self.fa.align(image, gray, rects[0].rect)

        return faceAligned


if __name__ == '__main__':
    face = FaceData()

    out = face.normalize('tmp/P012_Fear_023.jpeg')
    cv2.imwrite('tmp/foo_P012_Fear_023.jpeg',out)
