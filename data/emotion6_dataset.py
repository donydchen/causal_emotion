#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
from data.base_dataset import BaseDataset, get_transfroms
from data.image_folder import make_dataset_by_conf
import os
import pickle
from PIL import Image


class Emotion6Dataset(BaseDataset):
    """docstring for Emotion6Dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--cnfnd_name', type=str, default='scene_labels.pkl', help='cnfnd dictionary file path.')
        parser.set_defaults(cnfnd_num=10, emo_num=2)
        return parser

    def __init__(self, opt):
        super(Emotion6Dataset, self).__init__(opt)
        self.conf_name = opt.train_conf_name if opt.isTrain else opt.test_conf_name

        # load all image paths
        self.img_paths = sorted(make_dataset_by_conf(
                            os.path.join(opt.dataroot, 'images'),
                            os.path.join(opt.dataroot, self.conf_name),
                            opt.max_dataset_size,
                            name_to_path=self._parse_path_by_name))

        # load label dictionary
        with open(os.path.join(opt.dataroot, opt.emo_name), 'rb') as f:
            self.emo_dict = pickle.load(f)

        # load cnfnd dictionary
        with open(os.path.join(opt.dataroot, opt.cnfnd_name), 'rb') as f:
            self.cnfnd_dict = pickle.load(f)

        # used labels
        self.EMO_LABELS = ['Negative', 'Positive']
        self.CNFND_LABELS = ['beauty_salon', 'veterinarians_office', 'hospital_room', 'martial_arts_gym', 'shower',
                            'wheat_field', 'cemetery', 'stage/indoor', 'indoor', 'outdoor']

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        # load image
        img = Image.open(img_path).convert('RGB')
        img_transform = get_transfroms(self.opt)
        img = img_transform(img)

        # load image labels
        emo = self._parse_emo_by_path(img_path)
        # load confounder labels
        cnfnd = self._parse_cnfnd_by_path(img_path)

        return {'img': img, 'emo': emo, 'cnfnd': cnfnd, 'img_path': img_path}

    def __len__(self):
        return len(self.img_paths)

    def _parse_path_by_name(self, img_name):
        img_path = img_name
        return img_path

    def _parse_key_by_path(self, img_path):
        path_list = img_path.strip('/').split('/')
        img_key = ('_'.join(path_list[3:])).split('.')[0]
        return img_key

    def _parse_emo_by_path(self, img_path):
        img_key = self._parse_key_by_path(img_path)
        emo_label = self.emo_dict[img_key]
        if emo_label not in self.EMO_LABELS:
            raise Exception("The emotion is not defined, info: [%s, %s]" % (img_path, emo_label))
        return self.EMO_LABELS.index(emo_label)

    def _parse_cnfnd_by_path(self, img_path):
        img_key = self._parse_key_by_path(img_path)
        img_cnfnd = self.cnfnd_dict[img_key]
        if img_cnfnd not in self.CNFND_LABELS:
            raise Exception("The confounder is not defined, info: [%s, %s]" % (img_path, img_cnfnd))
        return self.CNFND_LABELS.index(img_cnfnd)
