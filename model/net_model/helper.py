import torch
import torchvision.transforms as transforms
import numpy as np
from torch.autograd.variable import Variable
import os

import torch
import torch.nn as nn


transform = transforms.ToTensor()



def compute_accuracy(prob_cls, gt_cls):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # we only need the detection which >= 0
    mask = torch.ge(gt_cls, 0)
    # get valid element
    valid_gt_cls = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls, 0.6).float()
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))


def convert_image_to_tensor(image):
    """convert an image to pytorch tensor
        Parameters:
        ----------
        image: numpy array , h * w * c
        Returns:
        -------
        image_tensor: pytorch.FloatTensor, c * h * w
        """
    image = image.astype(np.float)
    return transform(image)
    # return transform(image)


def convert_chwTensor_to_hwcNumpy(tensor):
    """convert a group images pytorch tensor(count * c * h * w) to numpy array images(count * h * w * c)
            Parameters:
            ----------
            tensor: numpy array , count * c * h * w
            Returns:
            -------
            numpy array images: count * h * w * c
            """

    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0, 2, 3, 1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0, 2, 3, 1))
    else:
        raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")





class CheckPoint(object):
    """
    save model state to file
    check_point_params: model, optimizer, epoch
    """

    def __init__(self, save_path):

        self.save_path = os.path.join(save_path)
        self.check_point_params = {'model': None,
                                   'optimizer': None,
                                   'epoch': None}

        # make directory
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def load_state(self, model, state_dict):
        """
        load state_dict to model
        :params model:
        :params state_dict:
        :return: model
        """
        model.eval()
        model_dict = model.state_dict()

        for key, value in list(state_dict.items()):
            if key in list(model_dict.keys()):
                # print key, value.size()
                model_dict[key] = value
            else:
                pass
                # print "key error:", key, value.size()
        model.load_state_dict(model_dict)
        # model.load_state_dict(state_dict)
        # set the model in evaluation mode, otherwise the accuracy will change
        # model.eval()
        # model.load_state_dict(state_dict)

        return model

    def load_model(self, model_path):
        """
        load model
        :params model_path: path to the model
        :return: model_state_dict
        """
        if os.path.isfile(model_path):
            print("|===>Load retrain model from:", model_path)
            # model_state_dict = torch.load(model_path, map_location={'cuda:1':'cuda:0'})
            model_state_dict = torch.load(model_path, map_location='cpu')
            return model_state_dict
        else:
            assert False, "file not exits, model path: " + model_path

    def load_checkpoint(self, checkpoint_path):
        """
        load checkpoint file
        :params checkpoint_path: path to the checkpoint file
        :return: model_state_dict, optimizer_state_dict, epoch
        """
        if os.path.isfile(checkpoint_path):
            print("|===>Load resume check-point from:", checkpoint_path)
            self.check_point_params = torch.load(checkpoint_path)
            model_state_dict = self.check_point_params['model']
            optimizer_state_dict = self.check_point_params['optimizer']
            epoch = self.check_point_params['epoch']
            return model_state_dict, optimizer_state_dict, epoch
        else:
            assert False, "file not exits" + checkpoint_path

    def save_checkpoint(self, model, optimizer, epoch, index=0):
        """
        :params model: model
        :params optimizer: optimizer
        :params epoch: training epoch
        :params index: index of saved file, default: 0
        Note: if we add hook to the grad by using register_hook(hook), then the hook function
        can not be saved so we need to save state_dict() only. Although save state dictionary
        is recommended, some times we still need to save the whole model as it can save all
        the information of the trained model, and we do not need to create a new network in
        next time. However, the GPU information will be saved too, which leads to some issues
        when we use the model on different machine
        """

        # get state_dict from model and optimizer
        model = self.list2sequential(model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.state_dict()
        optimizer = optimizer.state_dict()

        # save information to a dict
        self.check_point_params['model'] = model
        self.check_point_params['optimizer'] = optimizer
        self.check_point_params['epoch'] = epoch

        # save to file
        torch.save(self.check_point_params, os.path.join(
            self.save_path, "checkpoint_%03d.pth" % index))
        print("saving check_point")

    def list2sequential(self, model):
        if isinstance(model, list):
            model = nn.Sequential(*model)
        return model

    def save_model(self, model, best_flag=False, index=0, tag=""):
        """
        :params model: model to save
        :params best_flag: if True, the saved model is the one that gets best performance
        """
        # get state dict
        print("saving model")
        model = self.list2sequential(model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.state_dict()
        if best_flag:
            if tag != "":
                torch.save(model, os.path.join(self.save_path, "%s_best_model.pth" % tag))
            else:
                torch.save(model, os.path.join(self.save_path, "best_model.pth"))
        else:
            if tag != "":
                # torch.save(model, os.path.join(self.save_path, "model_%03d_%s.pth" % (index, tag)))
                torch.save(model, os.path.join(self.save_path, "model_%03d_%s.pth" % (index, tag)))
            else:
                torch.save(model, os.path.join(self.save_path, "model_%03d.pth" % index))
                # print(os.path.join(self.save_path, "model_%03d.pth" % index))