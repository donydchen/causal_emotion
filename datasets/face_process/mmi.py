#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
import os
from glob import glob
from align_faces import FaceData
import cv2
import pickle
from imutils import rotate_bound
import xml.etree.ElementTree as ET


FE_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']


def parse_frames(videos_dir, saved_dir):
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    subject_folders = sorted([x for x in glob(os.path.join(videos_dir, "*")) if os.path.isdir(x)])
    for subject in subject_folders:
        cur_video = glob(os.path.join(subject, "*.avi"))
        assert len(cur_video) == 1, "%s has %d video(s), not equal to 1." % (subject, len(cur_video))

        cur_video = cur_video[0]
        basename = os.path.basename(cur_video)
        saved_name_prefix = '_'.join(os.path.splitext(basename)[0].split('-'))

        vidcap = cv2.VideoCapture(cur_video)

        video_clips = []
        success, image = vidcap.read()
        while success:
            video_clips.append(image)
            success, image = vidcap.read()
        # print(len(video_clips))
        if len(video_clips) < 3:
            print("[ERR] Parsing: %s" % cur_video)
            continue

        center_num = int(len(video_clips) / 2)
        saved_clips = [center_num - 1, center_num, center_num + 1]
        for saved_idx in saved_clips:
            cur_name = "%s_%03d.png" % (saved_name_prefix, saved_idx)
            print(cur_name)
            cv2.imwrite(os.path.join(saved_dir, cur_name), video_clips[saved_idx])

    print('[END]Successfully parse and save all specific frames.')


def preprocess_imgs(in_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    right_subjects = set(['S001', 'S002', 'S017'])
    left_subjects = set(['S003', 'S005', 'S006', 'S015', 'S016'])
    normal_subjects = set(['S053', 'S054'])
    remove_subjects = set(['S021'])

    imgs_path = sorted(glob(os.path.join(in_dir, "*.png")))
    for img_path in imgs_path:
        cur_subject = os.path.basename(img_path).split('_')[0]
        if cur_subject in remove_subjects:
            continue

        print(img_path)
        saved_path = os.path.join(out_dir, os.path.basename(img_path))
        cur_img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        # width, height = cur_img.size
        height, width, _ = cur_img.shape
        if cur_subject in right_subjects:
            crop_img = cur_img[:height, int(width / 2):width]
        elif cur_subject in left_subjects:
            crop_img = cur_img[:height, :int(width / 2)]
        elif cur_subject in normal_subjects:
            crop_img = cur_img
        else:   # rotate -90
            crop_img = rotate_bound(cur_img, 90)
        cv2.imwrite(saved_path, crop_img)

    print('[END]Successfully pre-process all images.')


def align_face(in_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    face_data = FaceData()
    all_imgs = sorted(glob(os.path.join(in_dir, "*.png")))
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


def load_emotion_labels(videos_dir):
    subject_folders = sorted([x for x in glob(os.path.join(videos_dir, "*")) if os.path.isdir(x)])
    emotion_dict = {}

    for subject in subject_folders:
        cur_video = glob(os.path.join(subject, "*.avi"))
        assert len(cur_video) == 1, "%s has %d video(s), not equal to 1." % (subject, len(cur_video))
        cur_xml = "%s.xml" % os.path.splitext(cur_video[0])[0]
        assert os.path.isfile(cur_xml), "Can't find annotation file: %s" % cur_xml

        tree = ET.parse(cur_xml)
        root = tree.getroot()
        xml_emotion_node = root[1].attrib
        assert xml_emotion_node['Name'] == 'Emotion', "Error node provided for %s" % cur_xml
        cur_label = int(xml_emotion_node['Value'])

        if cur_label <= 6:
            cur_id = os.path.splitext(os.path.basename(cur_xml))[0].replace('-', "_")
            assert cur_id not in emotion_dict, "ID already exists, %s" % cur_xml
            cur_label = FE_LABELS[cur_label - 1]  # start from 0
            print(cur_id, cur_label)
            emotion_dict[cur_id] = cur_label

    print("[END]Successfully parse all labels. Sequence len:", len(emotion_dict.keys()))
    return emotion_dict


def update_images_and_labels(img_dir, raw_emotion_dict, out_path):
    all_imgs = sorted([os.path.basename(x) for x in glob(os.path.join(img_dir, "*.png"))])
    print(len(all_imgs))
    new_emotion_dict = {}
    removed_imgs = []
    for img in all_imgs:
        cur_key = "_".join(img.split("_")[:2])
        if cur_key in raw_emotion_dict:
            cur_label = raw_emotion_dict[cur_key]
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
    videos_dir = os.path.join(root_dir, "Sessions")
    frames_dir = os.path.join(root_dir, "mid_three_frames")
    # parse_frames(videos_dir, frames_dir)

    raw_dir = os.path.join(root_dir, "mid_three_raw_images")
    # preprocess_imgs(frames_dir, raw_dir)

    imgs_dir = os.path.join(new_out_dir, "images")
    # align_face(raw_dir, imgs_dir)

    raw_emotion_dict = load_emotion_labels(videos_dir)
    update_images_and_labels(imgs_dir, raw_emotion_dict, os.path.join(new_out_dir, "emotion_labels.pkl"))


if __name__ == '__main__':
    root_dir = 'datasets/MMI'
    new_out_dir = 'datasets/MixBasic/MMI'
    main(root_dir, new_out_dir)
