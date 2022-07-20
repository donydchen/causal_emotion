#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
import os
from glob import glob
from shutil import copyfile
from align_faces import FaceData
import cv2
import pickle


FE_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']


def filter_imgs(img_dir, out_dir):
    """
        filter the last three images out for further usage
    """
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    needed_imgs = []
    subjects = sorted(glob(os.path.join(img_dir, "*/")))
    for sub in subjects:
        series = sorted(glob(os.path.join(sub, "*/")))
        for ser in series:
            cur_imgs = sorted(glob(os.path.join(ser, "*.png")))
            needed_imgs.extend(cur_imgs[-3:])

    for img_path in needed_imgs:
        basename = os.path.basename(img_path)
        print("%s ==> %s" % (img_path, os.path.join(out_dir, basename)))
        copyfile(img_path, os.path.join(out_dir, basename))


def align_face(in_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    face_data = FaceData()
    all_imgs = sorted(glob(os.path.join(in_dir, "*.png")))
    for idx, img_path in enumerate(all_imgs):
        print("[%d/%d] Aligning %s..." % (idx, len(all_imgs), img_path))
        out_im = face_data.normalize(img_path)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), out_im)


def load_emotion_labels(labels_dir):
    emotion_dict = {}
    subjects = sorted(glob(os.path.join(labels_dir, "*/")))
    for sub in subjects:
        series = sorted(glob(os.path.join(sub, "*/")))
        for ser in series:
            # get label file
            label_file = sorted(glob(os.path.join(ser, "*.txt")))
            assert len(label_file) <= 1, "More than 1 label file: %s" % ser
            if len(label_file) == 0:
                continue
            label_file = label_file[0]
            # read emotion label from file
            with open(label_file, 'r') as f:
                lines = f.readlines()
                lines = [x.strip() for x in lines if x.strip()]
            assert len(lines) == 1, "Not just 1 label: %s" % label_file
            cur_label = FE_LABELS[int(float(lines[0]) - 1)]
            if cur_label == "Contempt":  # ignore contempt
                continue
            # store emotion
            cur_id = "_".join(os.path.basename(label_file).split("_")[:2])
            assert cur_id not in emotion_dict, "Duplicate image id: %s" % label_file
            emotion_dict[cur_id] = cur_label

    return emotion_dict


def update_images_and_labels(img_dir, emotion_dict, out_path):
    all_imgs = sorted([os.path.basename(x) for x in glob(os.path.join(img_dir, "*.png"))])
    print(len(all_imgs))
    new_emotion_dict = {}
    removed_imgs = []
    for img in all_imgs:
        cur_key = "_".join(img.split("_")[:2])
        if cur_key in emotion_dict:
            cur_label = emotion_dict[cur_key]
            new_emotion_dict[os.path.splitext(img)[0]] = cur_label
        else:
            removed_imgs.append(img)

    print("R: %d, K: %d" % (len(removed_imgs), len(list(new_emotion_dict.keys()))))
    # save emotion dict to pkl
    with open(out_path, 'wb') as f:
        pickle.dump(new_emotion_dict, f)
    for img_path in removed_imgs:
        full_imgpath = os.path.join(img_dir, img_path)
        print(full_imgpath)
        os.remove(full_imgpath)


def main(root_dir, new_out_dir):
    ori_dir = os.path.join(root_dir, "cohn-kanade-images")
    raw_dir = os.path.join(root_dir, "last_three_raw_images")
    # filter_imgs(ori_dir, raw_dir)

    img_dir = os.path.join(new_out_dir, "images")
    # align_face(raw_dir, img_dir)

    labels_dir = os.path.join(root_dir, "Emotion")
    emotion_dict = load_emotion_labels(labels_dir)

    out_path = os.path.join(new_out_dir, "emotion_labels.pkl")
    update_images_and_labels(img_dir, emotion_dict, out_path)


if __name__ == '__main__':
    root_dir = 'datasets/CKPlus'
    new_out_dir = 'datasets/MixBasic/CKPlus'
    main(root_dir, new_out_dir)
