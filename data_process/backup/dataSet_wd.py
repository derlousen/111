from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
# from torch.utils.serialization import load_lua
import pickle
import torch


class EvDataset(Dataset):

    def __init__(self, transform, is_train):
        # super(EvDataset, self).__init__(**kwargs)
        self.prefix = 'c:/dms_data/wd_face_gen/'
        f = open(self.prefix + 'data_tr.p', 'rb')
        self.data_list = pickle.load(f)
        f.close()

        self.img_list = self.data_list['img']
        self.label_list = self.data_list['label']

        self.data_len = len(self.img_list)

        self.transform = transform
        self.is_train = is_train
        self.show_count = 0

        del self.data_list

    def __getitem__(self, index):
        img, label = self.get_data(index)
        # label = np.array(label)
        label_bbox = label[1]

        if label[0] == 1:
            label_cls = torch.FloatTensor([1, 0])
        elif label[0] == -1:
            label_cls = torch.FloatTensor([0, 1])
            label_bbox = torch.FloatTensor([0, 0, 0, 0])
        else:
            # print(label[0])
            if label[0] == 0:
                label[0] = [0]
            # label_cls = torch.FloatTensor([label[0][0], 0])
            label_cls = torch.FloatTensor([0, 1])

        label_bbox = torch.FloatTensor(label_bbox)
        label_cls = torch.FloatTensor(label_cls)

        # sample = date_pre.window_gen(img)
        # out_img, offset = date_pre.get_bbox_offset(img, bbox)
        # print(label)

        # print([label_cls, label_bbox])
        return img, [label_cls, label_bbox]

    def __len__(self):
        return self.data_len

    def get_data(self, data_index):
        # print(self.prefix+self.data_list[data_index][0]+'.jpg')
        # img = cv2.imread(self.prefix + self.data_list[data_index][0] + '.jpg')
        # label = self.data_list[data_index][1:]
        img = self.img_list[data_index]
        # temp_img = img.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # img = cv2.Canny(img, 20, 30)
        #
        # # img = img[:, :, np.newaxis]
        # temp_img[:, :, 0] = img
        # temp_img[:, :, 1] = img
        # temp_img[:, :, 2] = img
        #
        # img = temp_img

        label = self.label_list[data_index]

        # self.show_count += 1
        # if self.show_count == 1280:
        #     self.show_count = 0
        #     cv2.imshow('img', cv2.resize(img, (0, 0), fx=8, fy=8))
        #
        #     k = cv2.waitKey(1)
        #     if k == 27:
        #         cv2.destroyAllWindows()

        img = np.array(img).astype(np.float)
        # print(img.shape)
        img = self.transform(img).float() / 255 - 0.5
        # print(np.max(img.data.numpy()))

        return img, label


if __name__ == "__main__":
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    )
    data_processor = EvDataset(transform1, None)
    data_processor.__getitem__(1000)
