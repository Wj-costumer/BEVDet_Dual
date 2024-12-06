# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D, BEVDet4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from copy import deepcopy

@DETECTORS.register_module()
class BEVStereo4D_Dual(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 **kwargs):
        super(BEVStereo4D_Dual, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        # self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.voxel_pool = nn.MaxPool1d(kernel_size=16)
        
        for name, param in self.named_parameters():
            if 'img_' in name:
                param.requires_grad = False
        print("=====DEBUG=====: finish build model")

    def loss_single(self,voxel_semantics,mask_camera,preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        torch.save(img_feats, "./debug_logs/img_feats_0.pt")
        # 后面多帧预测的时候需要修改一下
        results_list = [dict() for _ in range(len(img_metas))]
       
        # bev det head
        b, c, z, x, y = img_feats[0].shape
        # voxel_feats = img_feats[0].permute(0, 1, 3, 4, 2).reshape(-1, z)
        # bev_feats = self.voxel_pool(voxel_feats).reshape(b, c, x, y)
        bev_feats = img_feats[0].permute(0, 1, 3, 4, 2)[:, :, :, :, -1]
        
        
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        torch.save(occ_pred, './debug_logs/occ_pred_0.pt')
        # breakpoint()
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        bbox_pts = self.simple_test_pts([bev_feats], img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(results_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        result_dict['occ_res'] = occ_res
        return [result_dict]
        # return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # optim_params = []
        # for name, param in self.named_parameters():
        #     # if 'pts_bbox_head' not in name:
        #     if param.requires_grad == True:
        #         optim_params.append(name)
        #     if name == 'img_backbone.conv1.weight':
        #         print("debug: ", param[0, 0, :2, :2])
        # breakpoint()
        
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        # breakpoint()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        # breakpoint()
        # occ pred head
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        # breakpoint()
        # bev det head
        b, c, z, x, y = img_feats[0].shape
        bev_feats = img_feats[0].permute(0, 1, 3, 4, 2)[:, :, :, :, -1]
        # bev_feats = self.voxel_pool(voxel_feats) # .reshape(b, c, x, y)
        losses_pts = self.forward_pts_train([bev_feats], gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        # breakpoint()
        return losses
