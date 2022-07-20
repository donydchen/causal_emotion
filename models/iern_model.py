#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 19 Jul, 2022
# @Author  : Yuedong Chen (donydchen@gmail.com)
# @Link    : github.com/donydchen
import torch
from .base_model import BaseModel
from . import networks


class IERNModel(BaseModel):
    """docstring for IERNModel"""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--n_gen_blk', type=int, default=2, help='# of feature generator blocks')
        parser.add_argument('--n_recons_blk', type=int, default=3, help='# of reconstruction network blocks')
        parser.add_argument('--n_dis_blk', type=int, default=3, help='# of expression discriminator blocks')
        parser.add_argument('--lambda_cnfnd', type=float, default=5e-4, help='coefficient of center loss for domain feature')

        # parser.set_defaults()

        return parser

    def __init__(self, opt):
        super(IERNModel, self).__init__(opt)

        self.loss_names = ['ConGen', 'EmoGen', 'Recons', 'ConDis', 'EmoDis', 'Cnfnd', 'Cls']
        self.visual_names = ['input_img']
        self.model_names = ['Base', 'EmoGen', 'ConGen', 'Recons', 'EmoDis', 'ConDis', 'Cnfnd', 'Cls']

        # init network model
        self.netBase, self.netEmoGen, self.netConGen, self.netRecons, self.netEmoDis, self.netConDis, self.netCnfnd, self.netCls = \
            networks.define_iern(opt.emo_num, opt.cnfnd_num, 512, 7, opt.n_gen_blk, opt.n_recons_blk, opt.n_dis_blk,
                                    opt.backbone_archi, not opt.no_backbone_pretrained, opt.init_type, opt.init_gain, opt.gpu_ids)

        # init loss function and optimizers
        if self.isTrain:
            self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.criterionCenter = networks.CenterLoss(self.opt.cnfnd_num, 512 * 7 * 7).to(self.device)

            dis_params = [{'params': self.netEmoDis.parameters()},
                            {'params':self.netConDis.parameters()}]
            self.optimizer_Dis = torch.optim.Adam(dis_params, lr=opt.lr, betas=(opt.beta1, 0.999))

            feat_params = [{'params':self.netEmoGen.parameters()},
                            {'params':self.netConGen.parameters()},
                            {'params':self.criterionCenter.parameters()},
                            {'params':self.netRecons.parameters()}]
            self.optimizer_Feat = torch.optim.Adam(feat_params, lr=opt.lr, betas=(opt.beta1, 0.999))

            cls_params = [{'params': self.netBase.parameters(), 'lr': opt.base_lr},
                            {'params': self.netEmoGen.parameters()},
                            {'params': self.netRecons.parameters(), 'lr': 0.},  # no need to update reconstruction net in this step
                            {'params':self.netCls.parameters()}]
            self.optimizer_Cls = torch.optim.Adam(cls_params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = [self.optimizer_Dis, self.optimizer_Feat, self.optimizer_Cls]

    def set_input(self, input):
        self.input_img = input['img'].to(self.device)
        self.emo_label = input['emo'].type(torch.LongTensor).to(self.device)
        self.cnfnd_label = input['cnfnd'].type(torch.LongTensor).to(self.device)
        self.img_path = input['img_path']

        self.mean_emo_label = torch.ones([self.emo_label.size()[0], self.opt.emo_num], device=self.device) * (1. / self.opt.emo_num)
        self.mean_cnfnd_label = torch.ones([self.cnfnd_label.size()[0], self.opt.cnfnd_num], device=self.device) * (1. / self.opt.cnfnd_num)

    def forward(self):
        pass

    def forward_Feat(self):
        self.base_feat = self.netBase(self.input_img)
        self.emo_feat = self.netEmoGen(self.base_feat)
        self.con_feat = self.netConGen(self.base_feat)
        self.rec_base_feat = self.netRecons(self.emo_feat, self.con_feat)

    def forward_Cls(self):
        # update confounders
        self.confounders = self.netCnfnd(center_weights=self.criterionCenter.centers.detach())

        # forward one time for each confounder
        self.pred_emo = []
        for idx in range(self.opt.cnfnd_num):
            cur_confounder = self.confounders[idx, :, :, :]
            cur_confounder = cur_confounder.expand(self.emo_feat.size(0), -1, -1, -1)
            # combine expr feat with specific confounder
            cur_com_feat = self.netRecons(self.emo_feat, cur_confounder)
            cur_pred_emo = self.netCls(cur_com_feat)
            self.pred_emo.append(cur_pred_emo)

        self.pred_emo = torch.mean(torch.stack(self.pred_emo), dim=0)

    def forward_Test(self):
        self.base_feat = self.netBase(self.input_img)
        self.emo_feat = self.netEmoGen(self.base_feat)
        self.confounders = self.netCnfnd(is_test=True)  # load pretrained confounders

        # forward one time for each confounder
        pred_emo_list = []
        for idx in range(self.opt.cnfnd_num):
            cur_confounder = self.confounders[idx, :, :, :]
            cur_confounder = cur_confounder.expand(self.emo_feat.size(0), -1, -1, -1)
            cur_com_feat = self.netRecons(self.emo_feat, cur_confounder)
            cur_pred_emo = self.netCls(cur_com_feat)
            pred_emo_list.append(cur_pred_emo)

        self.pred_emo_all = torch.stack(pred_emo_list)
        self.pred_emo = torch.mean(self.pred_emo_all, dim=0)  # average across confounders

    def backward_Dis(self):
        self.dis_predict_emo = self.netEmoDis(self.emo_feat.detach())
        self.loss_EmoDis = self.criterionCE(self.dis_predict_emo, self.emo_label)

        self.dis_predict_cnfnd = self.netConDis(self.con_feat.detach())
        self.loss_ConDis = self.criterionCE(self.dis_predict_cnfnd, self.cnfnd_label)

        loss_Dis = (self.loss_EmoDis + self.loss_ConDis) / 2.
        loss_Dis.backward()

    def backward_Feat(self):
        self.dis_confused_emo = self.netEmoDis(self.con_feat)
        self.loss_ConGen = self.criterionMSE(torch.nn.functional.softmax(self.dis_confused_emo, dim=-1),
                                            self.mean_emo_label)

        self.dis_confused_cnfnd = self.netConDis(self.emo_feat)
        self.loss_EmoGen = self.criterionMSE(torch.nn.functional.softmax(self.dis_confused_cnfnd, dim=-1),
                                            self.mean_cnfnd_label)

        loss_Gen = (self.loss_EmoGen + self.loss_ConGen) / 2.

        self.loss_Cnfnd = self.criterionCenter(self.con_feat, self.cnfnd_label) * self.opt.lambda_cnfnd

        self.loss_Recons = self.criterionMSE(self.rec_base_feat, self.base_feat)

        loss_Feat = loss_Gen + self.loss_Cnfnd + self.loss_Recons
        loss_Feat.backward(retain_graph=True)  # Emotion feature still needed to be updated in backward_Cls
        # remove the effect of alpha on updating centers
        if self.opt.lambda_cnfnd != 0:
            for param in self.criterionCenter.parameters():
                param.grad.data = param.grad.data * (1. / self.opt.lambda_cnfnd)

    def backward_Cls(self):
        self.loss_Cls = self.criterionCE(self.pred_emo, self.emo_label)
        self.loss_Cls.backward()

    def optimize_parameters(self):
        self.forward_Feat()

        self.set_requires_grad(self.netEmoDis, True)
        self.set_requires_grad(self.netConDis, True)
        self.optimizer_Dis.zero_grad()
        self.backward_Dis()
        self.optimizer_Dis.step()

        self.set_requires_grad(self.netEmoDis, False)
        self.set_requires_grad(self.netConDis, False)
        self.optimizer_Feat.zero_grad()
        self.backward_Feat()
        self.optimizer_Feat.step()

        self.forward_Cls()

        self.optimizer_Cls.zero_grad()
        self.backward_Cls()
        self.optimizer_Cls.step()
