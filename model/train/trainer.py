import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
import os
import shelve

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import model.net_model.model as net_model
import torch
import model.train.helper as helper

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled == True
# torch.backends.cudnn.deterministic = True

from torchvision import transforms
# import random
import data_process.dataSet_heatmap_only as data_ev
from model.net_model.helper import CheckPoint
import model.train.config as train_config
import numpy as np

# Get config
config = train_config.Config()
config.save_path = "./results/onet/heatmap_only/"
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)


use_cuda = config.use_cuda and torch.cuda.is_available()
# torch.manual_seed(config.manualSeed)
# torch.cuda.manual_seed(config.manualSeed)
device = torch.device("cuda" if use_cuda else "cpu")


# Set dataloader
kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}

# f = open('/home/ev_ai/dms_data/pickle_gen/data_file_o.p', 'rb')
# data_list = pickle.load(f)
# data_list = data_list['label']
# random.shuffle(data_list)
# f.close()

# with shelve.open('/home/ev_ai/dms_data/key_point_data/data_file_3d.db') as f:
#     # b['total']=2
#     print(f['total'])


# Set model
# cuda3 = torch.device('cuda:3')

model_encoder = net_model.AttnEncoder()
model_decoder = net_model.AttnEncoder()

# model.load_state_dict(torch.load('model_039_0.33837378.pth'))

model_encoder_data = torch.nn.DataParallel(model_encoder, device_ids=[0]).cuda()
model_decoder_data = torch.nn.DataParallel(model_decoder, device_ids=[0]).cuda()
# model_data = model.to(device)

# Set checkpoint
checkpoint = CheckPoint(config.save_path)

# Set optimizer
optimizer_en = torch.optim.Adam(model_encoder_data.parameters(), lr=config.lr)
optimizer_de = torch.optim.Adam(model_decoder_data.parameters(), lr=config.lr)
# optimizer = torch.optim.Adagrad(model_data.parameters(), lr=config.lr)
# optimizer = torch.optim.Adagrad([
#         {'params': helper.get_parameters(model_data, bias=False)},
#         {'params': helper.get_parameters(model_data, bias=True),
#          'lr': config.lr * 0.01}
#     ], lr=config.lr)

# optimizer = torch.optim.RMSprop(model_data.parameters(), lr=config.lr)
# optimizer = torch.optim.SGD(model_data.parameters(), lr=config.lr, momentum=0.9)


scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_en, milestones=config.step, gamma=0.1)


data_list = 'data/'
data_seq1 = np.load(data_list+'bid.npy')
data_seq2 = np.load(data_list+'ask.npy')

data_len = len(data_seq1)
print(data_len)

offset = np.load('offset_2.np')

transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(
    data_ev.VcDataset(transform=transform, pickle_data=data_list, data_len=data_len, is_train=True, offset=offset),
    batch_size=config.batchSize, shuffle=False, **kwargs)


if __name__ == '__main__':
    import numpy as np

    trainer = train_config.PNetTrainer(config.lr, train_loader, model_data, optimizer, scheduler, device)

    for epoch in range(1, config.nEpochs + 1):
        cls_loss_, box_offset_loss, total_loss, accuracy = trainer.train(epoch)
        checkpoint.save_model(model, index=epoch, tag=str(trainer.total_loss))
        torch.cuda.empty_cache()
    result_log = np.array(trainer.result_log)
    # np.save('log.npy', result_log)
    for _ in result_log:
        print(_)
