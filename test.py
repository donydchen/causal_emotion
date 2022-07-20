#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
from util.util import cal_accuracy, cal_conf_mat, cal_per_accuracy
from tqdm import tqdm
import pickle
import os
from datetime import datetime


def get_cls_results(net_model, attr_name):
    tmp_cls = getattr(net_model, attr_name).cpu().float().numpy()
    if len(tmp_cls.shape) < 2:
        tmp_cls = np.expand_dims(tmp_cls, axis=0)

    tmp_cls = np.argmax(tmp_cls, axis=1).flatten()

    return list(tmp_cls)


if __name__ == '__main__':
    opt = TestOptions().parse()
    visualizer = Visualizer(opt)

    timestamp = datetime.now().strftime("%H%M%S")

    # update test options
    opt.serial_batches = True
    opt.preprocess = 'none'

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    pred_emo = []
    real_emo = []
    img_path = []

    train_ds_name = opt.checkpoints_dir.split('/')[1].lower()
    test_ds_name = opt.dataset_name.lower()

    with torch.no_grad():
        print("Start testing the model...")
        tqdm_bar = tqdm(total=min(opt.num_test, len(dataset)))
        for i, data in enumerate(dataset):
            if i * opt.batch_size >= opt.num_test:
                break
            tqdm_bar.update(opt.batch_size)

            model.set_input(data)
            model.forward_Test()
            # get predict label
            pred_emo += get_cls_results(model, "pred_emo")
            real_emo += list(model.emo_label.cpu().float().numpy())
            img_path += list(model.img_path)

        tqdm_bar.close()

    pred_emo = np.array(pred_emo).astype(int)
    real_emo = np.array(real_emo).astype(int)

    results_dict = {'pred_emo': pred_emo, 'real_emo': real_emo, 'img_path': img_path}

    # save log info
    with open(os.path.join(opt.results_dir,
            'logdata_%s_%s_%s.pkl' % (train_ds_name, test_ds_name, timestamp)), 'wb') as f:
        pickle.dump(results_dict, f)

    acc_num = cal_accuracy(real_emo, pred_emo)
    acc_dict = cal_per_accuracy(real_emo, pred_emo, labels_name=dataset.dataset.EMO_LABELS)
    acc_dict.update({'All': [acc_num, len(real_emo)]})
    visualizer.print_acc(acc_dict, prefix="[ACC][TRAIN:%s][TEST:%s][%s]" % (train_ds_name, test_ds_name, timestamp))

    conf_mat = cal_conf_mat(real_emo, pred_emo, labels_name=dataset.dataset.EMO_LABELS, normalize='true')
    # update confusion matrix name with postfix
    tmp_items = os.path.splitext(visualizer.cm_name)
    visualizer.cm_name = "%s_%s_%s_%s%s" % (tmp_items[0], train_ds_name, test_ds_name, timestamp, tmp_items[1])
    visualizer.draw_confusion_matrix(conf_mat, dataset.dataset.EMO_LABELS)

    print("[DONE] Results saved to %s." % opt.results_dir)
