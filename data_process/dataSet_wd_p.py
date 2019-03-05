from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
# from torch.utils.serialization import load_lua
import data_process.data_process_tool as data_pre
import torch
from queue import Queue
from threading import Thread


class VcDataset(Dataset):

    def __init__(self, pickle_data, data_len, transform, is_train, debug=0):
        # super(EvDataset, self).__init__(**kwargs)
        self.prefix = 'c:/dms_data/wd_face_gen/'
        # f = open(self.prefix + 'data_tr.p', 'rb')
        # self.data_list = pickle.load(f)
        # f.close()
        #
        # self.img_list = self.data_list['img']
        # self.label_list = self.data_list['label']
        #
        # self.data_len = len(self.img_list)

        self.data_list = pickle_data
        self.data_len = data_len

        self.data_ran_list = np.arange(0, data_len)
        np.random.shuffle(self.data_ran_list)

        self.transform = transform
        self.is_train = is_train
        self.show_count = 0

        self.show = debug
        self.q = Queue(maxsize=2048)

        if self.show:
            cv2.namedWindow("img", 0)
        # cv2.namedWindow('img', 0)

        t1 = Thread(target=self.read_pickle)
        t1.start()

    def read_pickle(self):
        while True:
            for i in range(self.data_len):
                # data_index = self.data_ran_list[i]
                data_index = str(self.data_ran_list[i])
                img = self.data_list[data_index][0]

                label = self.data_list[data_index][1:]

                rand_angle = np.random.uniform(-1, 1) * 40
                rand_flip = np.random.uniform(-1, 1)
                label_bbox = np.array(label[1])
                img, point = data_pre.rotate(img, np.zeros((1, 2)), rand_angle, label_bbox)
                if rand_flip > 0:
                    img, label_bbox = data_pre.flip_img_wd(img, label_bbox)

                label[1] = label_bbox

                if self.show:
                    self.show_count += 1
                    if self.show_count == 2048 * 2:
                        std_size = 24
                        show_img = img.copy()
                        pt1 = (int(std_size * label_bbox[0]), int(std_size * label_bbox[2]))
                        pt2 = (int(std_size * (1 + label_bbox[1])), int(std_size * (1 + label_bbox[3])))
                        # print(label_bbox)
                        # cv2.circle(show_img, pt1, 0, (0, 255, 0), 1)
                        # cv2.circle(show_img, pt2, 0, (0, 255, 0), 1)
                        cv2.rectangle(show_img, pt1, pt2, (0, 255, 0), 1)
                        self.show_count = 0
                        cv2.imshow('img', show_img)

                        k = cv2.waitKey(1)
                        if k == 27:
                            cv2.destroyAllWindows()

                img = np.array(img).astype(np.float)
                # print(img.shape)
                img = self.transform(img).float() / 127.5 - 1

                self.q.put([img, label])

    def __getitem__(self, index):
        img, label = self.q.get()
        # img, label = self.get_data(index)
        # print(img)
        label = np.array(label)
        label_bbox = label[1]

        x1, x2, y1, y2 = label_bbox

        w = (x2 - x1) < 0
        h = (y2 - y1) < 0
        small_size = w and h

        x_in = x1 > 0 and x2 < 0
        y_in = y1 > 0 and y2 < 0

        all_in = x_in and y_in

        approve = small_size or all_in

        if all_in:
            label[0] = 0.61

        # print(label_bbox, small_size, x_in, y_in, approve)

        if label[0] >= 0.60:
            label_cls = torch.FloatTensor([1, 0])
        elif label[0] >= 0.3:
            # iou = ((label[0]-0.1)*2)[0]
            label_cls = torch.FloatTensor([0, 1])
            # label_cls = torch.FloatTensor([iou, 1-iou])
            # label_bbox = torch.FloatTensor([0, 0, 0, 0])
        else:
            # print(label[0])

            label_cls = torch.FloatTensor([0, 1])
            label_bbox = torch.FloatTensor([0, 0, 0, 0])
            # if label[0] == 0:
            #     label[0] = [0]
            # # label_cls = torch.FloatTensor([label[0][0], 0])
            # label_cls = torch.FloatTensor([0, 1])

        label_bbox = torch.FloatTensor(label_bbox)
        label_cls = torch.FloatTensor(label_cls)

        # sample = date_pre.window_gen(img)
        # out_img, offset = date_pre.get_bbox_offset(img, bbox)
        # print(label)

        # print([label_cls, label_bbox])
        return img, [label_cls, label_bbox]

    def __len__(self):
        return self.data_len

    # def get_data(self, data_index):
    #     data_index = str(self.data_ran_list[data_index])
    #
    #     # print(self.prefix+self.data_list[data_index][0]+'.jpg')
    #     # img = cv2.imread(self.prefix + self.data_list[data_index][0] + '.jpg')
    #     # label = self.data_list[data_index][1:]
    #     img = self.data_list[data_index][0]
    #
    #     label = self.data_list[data_index][1:]
    #
    #     rand_angle = np.random.uniform(-1, 1) * 10
    #     rand_flip = np.random.uniform(-1, 1)
    #     label_bbox = np.array(label[1])
    #     img, point = data_pre.rotate(img, np.zeros((1, 2)), rand_angle, label_bbox)
    #     if rand_flip > 0:
    #         img, label_bbox = data_pre.flip_img_wd(img, label_bbox)
    #
    #     label[1] = label_bbox
    #
    #     if self.show:
    #         self.show_count += 1
    #         if self.show_count == 2048*2:
    #             std_size = 24
    #             show_img = img.copy()
    #             pt1 = (int(std_size * label_bbox[0]), int(std_size * label_bbox[2]))
    #             pt2 = (int(std_size * (1+label_bbox[1])), int(std_size * (1+label_bbox[3])))
    #             # print(label_bbox)
    #             # cv2.circle(show_img, pt1, 0, (0, 255, 0), 1)
    #             # cv2.circle(show_img, pt2, 0, (0, 255, 0), 1)
    #             cv2.rectangle(show_img, pt1, pt2, (0, 255, 0), 1)
    #             self.show_count = 0
    #             cv2.imshow('img', show_img)
    #
    #             k = cv2.waitKey(1)
    #             if k == 27:
    #                 cv2.destroyAllWindows()
    #
    #     img = np.array(img).astype(np.float)
    #     # print(img.shape)
    #     img = self.transform(img).float() / 127.5 - 1
    #     # print(np.max(img.data.numpy()))
    #
    #     return img, label


if __name__ == "__main__":
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    )
    data_processor = VcDataset(transform1, None)
    data_processor.__getitem__(1000)
