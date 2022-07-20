#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
import time
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


class Visualizer(object):
    """docstring for Visualizer"""

    def __init__(self, opt):
        super(Visualizer, self).__init__()
        self.opt = opt
        plt.switch_backend('agg')
        plt.rcParams["font.family"] = "Liberation Serif"

        if opt.display_id > 0 and opt.isTrain:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(log_dir=self.opt.checkpoints_dir)

        # create a logging file to store training losses
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

        root_dir = opt.checkpoints_dir if opt.isTrain else opt.results_dir
        self.acc_name = os.path.join(root_dir, 'emo_acc.txt')
        self.cm_name = os.path.join(root_dir, 'conf_mat.png')

    def print_current_losses(self, epoch, iters, data_len, losses, t_comp, t_data, cur_lr=None):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '[epoch: %d, iters: %05d/%05d, time: %.3fs, data: %.3fs]' % (epoch, iters, data_len,
                        t_comp, t_data)
        if cur_lr is not None:
            message += '[%s]' % (', '.join([str(x) for x in cur_lr]))

        for k, v in losses.items():
            message += ' %s: %.7f' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def plot_current_losses(self, epoch, total_iters, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            total_iters(int)
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        # print(epoch + counter_ratio)
        for k, v in losses.items():
            self.summary_writer.add_scalar('loss/loss_%s' % k, v, total_iters)

    def print_acc(self, scores, prefix="[Acc]"):
        message = prefix
        for k, v in scores.items():
            cur_acc = float(v[0]) / v[1]
            message += ' %s: %.6f(%d/%d) |' % (k, cur_acc, v[0], v[1])
        print(message)
        with open(self.acc_name, "a+") as f:
            f.write('%s\n' % message)

    def display_current_results(self, visuals, paths, total_iters):
        for k, v in visuals.items():
            scale_img = (v[0].cpu().float() + 1) / 2.0
            self.summary_writer.add_image('img/%s' % k, scale_img, total_iters)

    def draw_confusion_matrix(self, conf_mat, labels):
        color_map = 'Reds'
        cm_labels = [x.upper()[:2] for x in labels]
        df_cm = pd.DataFrame(conf_mat, index=cm_labels, columns=cm_labels)
        plt.figure(figsize=(6,6))
        sn.heatmap(df_cm, annot=True, cmap=color_map, fmt='.1f', cbar=False, annot_kws={"size": 15})
        drawn_fig = plt.gcf()
        plt.xticks(np.arange(len(cm_labels)) + 0.5, cm_labels)
        plt.yticks(np.arange(len(cm_labels)) + 0.5, cm_labels, rotation=90, va="center")

        drawn_fig.savefig(self.cm_name, bbox_inches='tight', dpi=800)
        plt.close()
