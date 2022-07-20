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
            cur_imgs = sorted(glob(os.path.join(ser, "*.jpeg")))
            needed_imgs.extend(cur_imgs[-3:])

    for img_path in needed_imgs:
        new_basename = "_".join(img_path.split("/")[-3:])
        print("%s ==> %s" % (img_path, os.path.join(out_dir, new_basename)))
        copyfile(img_path, os.path.join(out_dir, new_basename))


def align_face(in_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    face_data = FaceData()
    all_imgs = sorted(glob(os.path.join(in_dir, "*.jpeg")))
    for idx, img_path in enumerate(all_imgs):
        try:
            out_im = face_data.normalize(img_path)
            new_name = "%s.png" % (os.path.splitext(os.path.basename(img_path))[0])
            cv2.imwrite(os.path.join(out_dir, new_name), out_im)
            print("[%d/%d] Successfully aligned %s." % (idx + 1, len(all_imgs), img_path))
        except Exception:
            print("====> Fail to detect face for %s." % (idx + 1, len(all_imgs), img_path))
            with open(os.path.join(out_dir, '../err.log'), 'a+') as f:
                f.write("%s\n" % img_path)


def build_emotion_labels(img_dir, out_path):
    all_imgs = sorted([os.path.basename(x) for x in glob(os.path.join(img_dir, "*.png"))])
    print(len(all_imgs))
    emotion_dict = {}
    for imgname in all_imgs:
        cur_label = imgname.split("_")[1]
        emotion_dict[os.path.splitext(imgname)[0]] = cur_label

    with open(out_path, 'wb') as f:
        pickle.dump(emotion_dict, f)


def main(root_dir, new_out_dir):
    ori_dir = os.path.join(root_dir, "VL", "Strong")
    raw_dir = os.path.join(root_dir, "last_three_raw_images")
    # filter_imgs(ori_dir, raw_dir)

    img_dir = os.path.join(new_out_dir, "images")
    # align_face(raw_dir, img_dir)

    build_emotion_labels(img_dir, os.path.join(new_out_dir, "emotion_labels.pkl"))


if __name__ == '__main__':
    root_dir = 'datasets/OuluCASIA'
    new_out_dir = 'datasets/MixBasic/OuluCASIA'
    main(root_dir, new_out_dir)
