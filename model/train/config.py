import torch
import datetime
import time
from model.net_model.lossfn import LossFn, wing_loss
from model.train.helper import AverageMeter
from tqdm import tqdm

class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        #  ------------ General options ----------------------------------------
        self.save_path = "./results/pnet/"
        self.dataPath = "'/home/dataset/WIDER/WIDER_train/images"  # path for loading data set
        self.annoPath = "./annotations/imglist_anno_12.txt"
        self.manualSeed = 1  # manually set RNG seed
        self.use_cuda = True
        self.GPU = "3"  # default gpu to use

        # ------------- Data options -------------------------------------------
        self.nThreads = 5  # number of data loader threads

        # ---------- Optimization options --------------------------------------
        self.nEpochs = 500  # number of total epochs to train 400
        self.batchSize = 32  # mini-batch size 128   ##Do not set to 1

        # lr master for optimizer 1 (mask vector d)
        self.lr = 3e-4  # initial learning rate
        self.step = [10, 25, 40]  # step for linear or exp learning rate policy
        self.decayRate = 0.1  # lr decay rate
        self.endlr = -1

        # ---------- Model options ---------------------------------------------
        self.experimentID = "2_batch"

        # ---------- Resume or Retrain options ---------------------------------------------
        self.resume = None  # "./checkpoint_064.pth"
        self.retrain = None

        self.save_path = self.save_path + "log_bs{:d}_lr{:.3f}_{}/".format(self.batchSize, self.lr, self.experimentID)


class PNetTrainer(object):

    def __init__(self, lr, train_loader, model, optimizer, scheduler, device):
        self.lr = lr
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lossfn = LossFn(self.device)
        # self.lossfn = wing_loss(self.device)
        # self.logger = None
        self.run_count = 0
        # self.scalar_info = {}
        # self.result_log=[]
        self.total_loss = 0

    def compute_accuracy(self, prob_cls, gt_cls):
        # we only need the detection which >= 0
        # print(prob_cls)

        prob_cls = torch.squeeze(prob_cls)

        mask = torch.ge(gt_cls, 0)
        # get valid element
        valid_gt_cls = torch.masked_select(gt_cls, mask)
        valid_prob_cls = torch.masked_select(prob_cls, mask)
        size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
        prob_ones = torch.ge(valid_prob_cls, 0.6).float()
        right_ones = torch.eq(prob_ones, valid_gt_cls.float()).float()

        return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, epoch):
        import sys

        cls_loss_ = AverageMeter()
        box_offset_loss_ = AverageMeter()
        total_loss_ = AverageMeter()
        accuracy_ = AverageMeter()

        # self.scheduler.step()
        self.model.train()
        total_num = 0
        # t1 = time.clock()

        total_sample = len(self.train_loader)
        pbar = tqdm(total=total_sample, ncols=75)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # print(sys.getsizeof(self))
            # print(sys.getsizeof(self.optimizer))
            # gt_label = target[0]
            # gt_bbox = target[1]
            # gt_point = target[0]
            # gt_cls = target[1]
            # gt_point_offset = target[2]
            gt_heatmap = target


            # data, gt_point, gt_cls, gt_point_offset, gt_heatmap = \
            #     data.to(self.device), gt_point.to(self.device).float(), gt_cls.to(self.device).float(), \
            #     gt_point_offset.to(self.device), gt_heatmap.to(self.device)

            data, gt_heatmap = \
                data.to(self.device), gt_heatmap.to(self.device)

            # cls_pred, box_offset_pred, point_pred = self.model(data)
            heatmap_pred = self.model(data)

            # print(cls_pred[0], gt_cls[0])

            # compute the loss
            # print(cls_pred)

            # cls_loss = self.lossfn.loss_cls(cls_pred, gt_cls)
            # offset_loss = self.lossfn.loss_offset(point_offset_pred, gt_point_offset)
            # point_loss = self.lossfn.loss_landmark(point_pred, gt_point)

            heatmap_loss = self.lossfn.loss_heatmap(heatmap_pred, gt_heatmap)

            # print('here')

            # total_loss = cls_loss * 0.3 + point_loss + offset_loss + heatmap_loss
            total_loss = heatmap_loss

            # total_loss = point_loss
            # print(cls_loss, box_offset_loss,point_loss, total_loss)
            # accuracy = self.compute_accuracy(cls_pred, gt_label)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # cls_loss_.update(cls_loss, data.size(0))
            # box_offset_loss_.update(box_offset_loss, data.size(0))
            show_loss = total_loss.cpu().data.numpy()
            total_loss_.update(show_loss, data.size(0))
            # accuracy_.update(accuracy, data.size(0))
            # torch.cuda.empty_cache()
            if total_num % 500 == 0:
                show_loss = total_loss.cpu().data.numpy()
                total_loss_.update(show_loss, data.size(0))
                pbar.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\tAccuracy: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), float(show_loss), float(show_loss)))

                # tqdm.write(time.time() - t1, '\t', (1 - batch_idx / len(self.train_loader)))
                # t1 = time.time()
            total_num += 1
            pbar.update(1)
        # self.scalar_info['cls_loss'] = cls_loss_.avg
        # self.scalar_info['box_offset_loss'] = box_offset_loss_.avg
        # self.scalar_info['total_loss'] = total_loss_.avg
        # self.scalar_info['accuracy'] = accuracy_.avg
        # self.scalar_info['lr'] = self.scheduler.get_lr()[0]
        #
        # if self.logger is not None:
        #     for tag, value in list(self.scalar_info.items()):
        #         self.logger.scalar_summary(tag, value, self.run_count)
        #     self.scalar_info = {}
        self.run_count = self.run_count + 1

        print("Epoch{} |===>Loss: {:.4f}".format(epoch, total_loss_.avg))
        self.total_loss = total_loss_.avg
        # self.result_log.append([epoch, total_loss_.avg])
        pbar.close()
        return cls_loss_.avg, box_offset_loss_.avg, total_loss_.avg, accuracy_.avg
