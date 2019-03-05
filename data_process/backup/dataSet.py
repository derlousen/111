from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import pickle
import torch


class EvDataset(Dataset):

    def __init__(self, transform, is_train):
        # super(EvDataset, self).__init__(**kwargs)
        self.prefix = 'c:/dms_data/'
        f = open(self.prefix + '12/data.p', 'rb')
        self.data_list = pickle.load(f)
        f.close()

        self.data_len = len(self.data_list)
        self.transform = transform
        self.is_train = is_train
        self.show_count = 0

    def __getitem__(self, index):
        img, label = self.get_data(index)
        # label = np.array(label)
        label_bbox = label[1:]
        if label[0] == 1:
            label_cls = torch.FloatTensor([1, 0])
        elif label[0] == -1:
            label_cls = torch.FloatTensor([0, 1])
            label_bbox = torch.FloatTensor([-1, -1, -1, -1])
        else:
            label_cls = torch.FloatTensor([0, 1])

        label_bbox = torch.FloatTensor(label_bbox)
        label_cls = torch.FloatTensor(label_cls)

        # sample = date_pre.window_gen(img)
        # out_img, offset = date_pre.get_bbox_offset(img, bbox)
        # print(label)

        return img, [label_cls, label_bbox]

    def __len__(self):
        return self.data_len

    def get_data(self, data_index):
        # print(self.prefix+self.data_list[data_index][0]+'.jpg')
        img = cv2.imread(self.prefix + self.data_list[data_index][0] + '.jpg')
        label = self.data_list[data_index][1:]

        # self.show_count += 1
        # if self.show_count == 1280:
        #     self.show_count = 0
        #     cv2.imshow('img', cv2.resize(img, (0, 0), fx=8, fy=8))
        #
        #     k = cv2.waitKey(1)
        #     if k == 27:
        #         cv2.destroyAllWindows()

        img = np.array(img).astype(np.float)
        img = self.transform(img).float()/127.5-1.0
        # print(img)

        return img, label


if __name__ == "__main__":
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    )
    data_processor = EvDataset(transform1, None)
    data_processor.__getitem__(1000)
