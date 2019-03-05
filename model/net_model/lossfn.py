import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import sys
# sys.path.append('../../')
# sys.path.append('../')
# sys.path.append('./')
# from model.net_model.vdc_loss import VDCLoss

class LossFn:
    def __init__(self, device):
        # loss function
        self.loss_cls = nn.MSELoss().to(device)
        self.loss_offset = nn.MSELoss().to(device)
        self.loss_landmark = wing_loss(device).to(device)
        self.loss_heatmap = nn.MSELoss().to(device)


    # def cls_loss(self, gt_label, pred_label):
    #     # get the mask element which >= 0, only 0 and 1 can effect the detection loss
    #     # print(pred_label[0])
    #     pred_label = torch.squeeze(pred_label)
    #     mask = torch.ge(gt_label, 0.6)
    #     valid_gt_label = torch.masked_select(gt_label, mask).float()
    #     # print(mask,pred_label)
    #     valid_pred_label = torch.masked_select(pred_label, mask)
    #     return self.loss_cls(valid_pred_label, valid_gt_label)
    #
    # def box_loss(self, gt_label, gt_offset, pred_offset):
    #     # get the mask element which != 0
    #     mask = torch.ne(gt_label, 0)
    #     # convert mask to dim index
    #     chose_index = torch.nonzero(mask)
    #     chose_index = torch.squeeze(chose_index)
    #     # only valid element can effect the loss
    #     # print(gt_offset)
    #     valid_gt_offset = gt_offset[chose_index, :]
    #     valid_pred_offset = pred_offset[chose_index, :]
    #     valid_pred_offset = torch.squeeze(valid_pred_offset)
    #     # print(valid_pred_offset.shape, valid_gt_offset.shape)
    #     return self.loss_box(valid_pred_offset, valid_gt_offset)
    #
    # def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
    #     # mask = torch.eq(gt_label, 1)
    #     #
    #     # chose_index = torch.nonzero(mask.data)
    #     # chose_index = torch.squeeze(chose_index)
    #     #
    #     # valid_gt_landmark = gt_landmark[chose_index, :]
    #     # valid_pred_landmark = pred_landmark[chose_index, :]
    #
    #     valid_gt_landmark = gt_landmark
    #     valid_pred_landmark = pred_landmark
    #
    #     return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)


class wing_loss(nn.Module):
    def __init__(self, device, w=10.0, epsilon=2.0):
        super(wing_loss, self).__init__()
        self.w = torch.Tensor(np.array(w)).to(device)
        self.epsilon = torch.Tensor(np.array(epsilon)).to(device)
        self.device = device

    def forward(self, pred, truth):
        x = pred - truth
        x = x.to(self.device)
        c = self.w * (1.0 - torch.log(1.0 + self.w / self.epsilon))
        x = torch.abs(x)

        losses = torch.where(torch.gt(self.w, x),
                             self.w * torch.log(1.0 + x / self.epsilon), (x - c))

        losses = losses.mean()
        return losses
