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


class MixBasicDataset(BaseDataset):
    """docstring for MixBasicDataset"""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(emo_num=6, cnfnd_num=3)
        return parser

    def __init__(self, opt):
        super(MixBasicDataset, self).__init__(opt)
        conf_name = opt.train_conf_name if opt.isTrain else opt.test_conf_name

        # load all image paths
        self.img_paths = sorted(make_dataset_by_conf(
                            opt.dataroot,
                            os.path.join(opt.dataroot, conf_name),
                            opt.max_dataset_size,
                            name_to_path=self._parse_path_by_name))

        # load label dictionary
        with open(os.path.join(opt.dataroot, opt.emo_name), 'rb') as f:
            self.emo_dict = pickle.load(f)

        # used labels
        self.EMO_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
        self.CNFND_LABELS = ['CKPlus', 'MMI', 'OuluCASIA']

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        # load image
        img = Image.open(img_path).convert('RGB')
        img_transform = get_transfroms(self.opt)
        img = img_transform(img)

        # load image labels
        img_emo = self._parse_emo_by_path(img_path)
        if img_emo not in self.EMO_LABELS:
            raise Exception("The expression label is not defined, info: [%s, %s]" % (img_path, img_emo))
        emo = self.EMO_LABELS.index(img_emo)

        # load confounder labels
        img_cnfnd = self._parse_cnfnd_by_path(img_path)
        if img_cnfnd not in self.CNFND_LABELS:
            raise Exception("The confounder is not defined, info: [%s, %s]" % (img_path, img_cnfnd))
        cnfnd = self.CNFND_LABELS.index(img_cnfnd)

        return {'img': img, 'emo': emo, 'cnfnd': cnfnd, 'img_path': img_path}

    def __len__(self):
        return len(self.img_paths)

    def _parse_path_by_name(self, img_name):
        img_path = '/'.join(img_name.split('/')[1:])
        return img_path

    def _parse_emo_by_path(self, img_path):
        path_list = img_path.strip('/').split('/')
        img_key = "%s_%s" % (path_list[-3].lower(), os.path.splitext(path_list[-1])[0])
        return self.emo_dict[img_key]

    def _parse_cnfnd_by_path(self, img_path):
        path_list = img_path.strip('/').split('/')
        ds = path_list[-3]
        return ds
