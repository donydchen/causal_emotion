#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
import pickle
import os
from tkinter.ttk import Labelframe


root_dir = 'datasets/MixBasic'


all_labels = {}
subfolders = ['CKPlus', 'MMI', 'OuluCASIA']
for subfolder in subfolders:
    label_file = os.path.join(root_dir, subfolder, 'emotion_labels.pkl')
    with open(label_file, 'rb') as f:
        ori_dict = pickle.load(f)
    for k, v in ori_dict.items():
        all_labels['%s_%s' % (subfolder.lower(), k)] = v

with open(os.path.join(root_dir, 'emotion_labels.pkl'), 'wb') as f:
    pickle.dump(all_labels, f)

print('All Done!')
