from torch.utils.data import Dataset
import numpy as np
import cv2
# from torch.utils.serialization import load_lua
import data_process.data_process_tool as date_pre


class EvDataset(Dataset):

    def __init__(self, transform, is_train):
        # super(EvDataset, self).__init__(**kwargs)
        img_list = "../img.npy"
        label_list = '../label.npy'
        self.img_list = np.load(img_list)
        self.label_list = np.load(label_list)
        self.data_len = len(self.img_list)
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        img, bbox = self.get_data(index)
        # sample = date_pre.window_gen(img)

        out_img, offset = date_pre.get_bbox_offset(img, bbox)
        print(offset)

        pass

    def __len__(self):
        return self.data_len

    def get_data(self, data_index):
        label_data = load_lua(self.label_list[data_index], long_size=8)
        img = cv2.imread(self.img_list[data_index])

        img, bbox = date_pre.get_bbox(img, label_data)

        # cv2.imshow('img', img)
        # print(bbox)
        # k = cv2.waitKey(0)

        # if k == 27:
        #     cv2.destroyAllWindows()
        return img, bbox


if __name__ == "__main__":
    data_processor = EvDataset()
    data_processor.__getitem__(1000)
