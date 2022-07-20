#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import time
import torch
import numpy as np
import copy
from util.util import cal_accuracy, cal_conf_mat, cal_per_accuracy
from tqdm import tqdm
from test import get_cls_results


def test_model(model, dataset):
    model.eval()
    pred_emo = []
    real_emo = []

    with torch.no_grad():
        tqdm_bar = tqdm(total=len(dataset))
        for i, data in enumerate(dataset):
            tqdm_bar.update(len(data['img_path']))

            model.set_input(data)
            model.forward_Test()
            pred_emo += get_cls_results(model, "pred_emo")
            real_emo += list(model.emo_label.cpu().float().numpy())

        tqdm_bar.close()

    pred_emo = np.array(pred_emo).astype(int)
    real_emo = np.array(real_emo).astype(int)

    acc_num = cal_accuracy(real_emo, pred_emo)
    acc_dict = cal_per_accuracy(real_emo, pred_emo, labels_name=dataset.dataset.EMO_LABELS)
    acc_dict.update({'All': [acc_num, len(real_emo)]})
    conf_mat = cal_conf_mat(real_emo, pred_emo, labels_name=dataset.dataset.EMO_LABELS, normalize='true')

    return acc_dict, conf_mat


if __name__ == '__main__':
    opt = TrainOptions().parse()
    visualizer = Visualizer(opt)

    # build dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    # build model
    model = create_model(opt)
    model.setup(opt)

    # init test setting
    test_opt = copy.deepcopy(opt)
    test_opt.isTrain = False
    test_opt.serial_batches = True
    test_opt.preprocess = 'none'
    test_opt.dataroot = test_opt.test_dataroot
    test_opt.dataset_name = test_opt.test_dataset_name
    test_opt.batch_size = test_opt.test_batch_size
    test_dataset = create_dataset(test_opt)
    print('The number of validating images = %d' % len(test_dataset))

    total_iters = 0
    if opt.lr_policy == "warmup":
        n_epochs_end = opt.n_epochs + opt.n_epochs_decay + opt.n_epochs_warmup + 1
        # update learning rate to skip the first 0 case
        for optimizer in model.optimizers:
            optimizer.zero_grad()
            optimizer.step()
        model.update_learning_rate()
    else:
        n_epochs_end = opt.n_epochs + opt.n_epochs_decay + 1

    for epoch in range(opt.epoch_count, n_epochs_end):
        model.train()

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on tensorboard
                model.compute_visuals()
                if opt.display_id > 0:
                    visualizer.display_current_results(model.get_current_visuals(), model.get_image_paths(), total_iters)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time)   # per batch
                cur_lr = model.get_learning_rate()
                visualizer.print_current_losses(epoch, epoch_iter, dataset_size, losses, t_comp, t_data, cur_lr)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, total_iters, losses)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs_end - 1, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    # test the model
    test_start_t = time.time()
    acc_dict, conf_mat = test_model(model, test_dataset)
    print("Test time cost: %fs" % (time.time() - test_start_t))
    visualizer.print_acc(acc_dict)
    visualizer.draw_confusion_matrix(conf_mat, dataset.dataset.EMO_LABELS)

    # save the last epoch
    model.save_networks(n_epochs_end - 1)
