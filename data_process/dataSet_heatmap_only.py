from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
# from torch.utils.serialization import load_lua
import torch
import data_process.data_process_tool as data_pre


class VcDataset(Dataset):

    def __init__(self, pickle_data, data_len, transform, is_train, offset, debug=0):
        # super(EvDataset, self).__init__(**kwargs)
        # self.prefix = 'c:/dms_data/wd_face_gen/'
        # f = open(self.prefix + 'data_tr.p', 'rb')
        # self.data_list = pickle.load(f)
        # f.close()
        #
        # self.img_list = self.data_list['img']
        # self.label_list = self.data_list['label']
        #
        # self.data_len = len(self.img_list)

        self.offset = offset
        self.data_list = pickle_data
        self.data_len = data_len

        self.data_ran_list = np.arange(0, data_len)
        np.random.shuffle(self.data_ran_list)

        self.transform = transform
        self.is_train = is_train
        self.show_count = 0
        self.show = debug
        # self.point_index_ = [1, 9, 17, 37, 40, 43, 46, 31, 49, 55]

        if self.show:
            cv2.namedWindow("img", 0)

    def __getitem__(self, index):
        img, label = self.get_data(index)
        # label = np.array(label)

        # label_iou = label[0]
        # label_bbox = label[1]
        # pointlist = label[2]
        # point_offset = label[3]
        heatmap = label

        #  10 points, 20 values

        # point_index = []
        # for _ in self.point_index_:
        #     point_index.append(2 * (_ - 1))
        #     point_index.append(2 * (_ - 1) + 1)
        # pointlist = pointlist[point_index]

        # print(pointlist)
        # x1, x2, y1, y2 = label_bbox
        #
        # # w = (x2 - x1) < 0
        # # h = (y2 - y1) < 0
        # # small_size = w and h
        #
        # x_in = x1 > 0 and x2 < 0
        # y_in = y1 > 0 and y2 < 0
        #
        # all_in = x_in and y_in
        #
        # # approve = small_size or all_in
        #
        # if all_in:
        #     label_iou = 0.65
        #
        # if label_iou >= 0.4:
        #     label_cls = torch.FloatTensor([1, 0])
        #
        # else:
        #     label_cls = torch.FloatTensor([0, 1])

        # label_cls = torch.FloatTensor(label_cls)
        # pointlist = torch.FloatTensor(pointlist)
        # point_offset = torch.FloatTensor(point_offset)
        heatmap = torch.FloatTensor(heatmap)

        # sample = date_pre.window_gen(img)
        # out_img, offset = date_pre.get_bbox_offset(img, bbox)
        # print(label)

        # print([label_cls, label_bbox])
        # print(pointlist, point_offset)
        # print(torch.max(heatmap))
        return img, heatmap
        # return img, pointlist

    def __len__(self):
        return self.data_len

    def get_data(self, data_index):
        # print(self.data_ran_list[data_index])

        # print(self.prefix+self.data_list[data_index][0]+'.jpg')
        # img = cv2.imread(self.prefix + self.data_list[data_index][0] + '.jpg')
        # label = self.data_list[data_index][1:]
        img, iou, offset, pointlist = self.data_list[str(self.data_ran_list[data_index])]
        # img = cv2.resize(img, (96, 96))
        heatmap = np.zeros((96, 96, 6), dtype=np.uint8)

        points = np.array(pointlist * 96).astype(np.int)

        # face outline
        # for point in points:
        #     cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        temp = np.copy(heatmap[:, :, 0])
        for point in points[0:17]:
            cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        heatmap[:, :, 0] = temp

        # eyebrow
        temp = np.copy(heatmap[:, :, 1])
        for point in points[17:27]:
            cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        heatmap[:, :, 1] = temp

        # nose
        temp = np.copy(heatmap[:, :, 2])
        for point in points[27:31]:
            cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        for point in points[31:36]:
            cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        heatmap[:, :, 2] = temp

        ###################################################
        # eyes
        temp = np.copy(heatmap[:, :, 3])
        for point in points[36:42]:
            cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        heatmap[:, :, 3] = temp
        temp = np.copy(heatmap[:, :, 4])
        for point in points[42:48]:
            cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        heatmap[:, :, 4] = temp
        # mouth
        temp = np.copy(heatmap[:, :, 5])
        for point in points[49:68]:
            cv2.circle(temp, tuple(point), 0, (1, 1, 1), 3)
        heatmap[:, :, 5] = temp
        ###################################################
        heatmap = heatmap.transpose((2, 0, 1))
        heatmap = heatmap.astype(np.float)

        # rand_angle = np.random.uniform(-1, 1) * 40
        # rand_flip = np.random.uniform(-1, 1)
        # offset = np.array(offset)
        # img, pointlist = data_pre.rotate(img, pointlist, rand_angle, offset, 96)

        # if rand_flip > 0:
        #     img, pointlist = data_pre.flip_img(img, pointlist)

        # img = np.array(img).astype(np.float)

        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float)
        img = torch.Tensor(img.copy())/127.5 - 1

        # img = self.transform(img).float() / 127.5 - 1
        # print(np.max(img.data.numpy()))

        # pointlist = pointlist - self.offset
        #
        # point_offset = np.mean(pointlist, axis=0)
        #
        # pointlist[:, 0] = pointlist[:, 0] - point_offset[0]
        # pointlist[:, 1] = pointlist[:, 1] - point_offset[1]
        # # -----------------------------------------------------
        # pointlist = pointlist * 10.
        # # -----------------------------------------------------
        #
        # # print(np.mean(pointlist))
        # pointlist = pointlist.flatten()

        # if self.show:
        #     self.show_count += 1
        #     if self.show_count == 4096:
        #         self.show_count = 0
        #         show_img = img.numpy()
        #         show_img += 1
        #         show_img *= 127.5
        #
        #         show_img = show_img.transpose((1, 2, 0))
        #         show_img = show_img.astype(np.uint8)
        #         cv2.imshow('img', show_img)
        #
        #         k = cv2.waitKey(1)
        #         if k == 27:
        #             cv2.destroyAllWindows()

        # print(pointlist)
        return img, heatmap


if __name__ == "__main__":
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    )
    data_processor = VcDataset(transform1, None)
    data_processor.__getitem__(1000)
