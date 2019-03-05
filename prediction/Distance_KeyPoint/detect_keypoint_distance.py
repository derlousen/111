import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import torch

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled == True

import model.net_model.model as netmodel
import cv2
import numpy as np
from torchvision import transforms
import time
import prediction.helper as helper
import torch.nn as nn
from threading import Thread
import math
from torchsummary import summary


class FaceBBox(object):
    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cam_size = 1080

    if cam_size == 720:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, img = cap.read()
        cam_left_size = 280

    elif cam_size == 1080:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, img = cap.read()
        cam_left_size = 420

    if cam_size == 720:
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret2, img2 = cap2.read()
        cam_left_size2 = 280

    elif cam_size == 1080:
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret2, img2 = cap2.read()
        cam_left_size2 = 420

    show_img_bbx_from_detection = img.copy()

    detected = 0
    tracking_count = 0
    cls_r = 0
    tracking = 0

    cls_r_threshold = 0.2

    tracking_img = np.zeros((720, 720, 3))

    modelP = netmodel.PNet()
    modelR = netmodel.RNet()
    modelO = netmodel.ONet_cls()

    # point_index_ = [1, 9, 17, 37, 40, 43, 46, 31, 49, 55]
    # point_index = []
    # for _ in point_index_:
    #     point_index.append(2 * (_ - 1))
    #     point_index.append(2 * (_ - 1) + 1)
    point_off = np.load('offset.np')
    point_off = point_off.flatten()
    # point_off = point_off[point_index]

    img_size = 96

    running = 1
    new_img = 0

    keypoint_img_save = np.zeros((128, 128, 3), dtype=np.uint8)

    keypoint_size = 0
    keypoint_topleft = [0, 0]
    keypoint_detected = 0

    kal = 0
    kal_points = 0
    kal_cls = 0
    kal_distance = 0

    new_cam_img = np.zeros((cam_size, cam_size, 3), dtype=np.uint8)
    new_cam_img2 = np.zeros((cam_size, cam_size, 3), dtype=np.uint8)

    def __init__(self,
                 p_model='weight_P/model_152_0.13011982.pth',
                 r_model='weight_R/model_157_0.10040249.pth',
                 o_model='weight_O/model_206_0.3007328877101077.pth', ):

        self.modelP.cuda()
        self.modelR.cuda()
        self.modelO.cuda()

        summary(self.modelO, input_size=(3, 96, 96))

        self.modelP.eval()
        self.modelR.eval()
        self.modelO.eval()

        self.modelP.load_state_dict(torch.load(p_model))
        self.modelR.load_state_dict(torch.load(r_model))
        self.modelO.load_state_dict(torch.load(o_model))

        self.distance_img = 0

        t1 = Thread(target=self.run_net)
        t2 = Thread(target=self.run_cam)
        t1.start()
        t2.start()

    def run_cam(self):
        cv2.namedWindow("img_bbx_from_detection", 0)
        cv2.namedWindow("keypoint_img", 0)

        while self.running:
            ret, img = self.cap.read()
            self.new_cam_img = self.resize_cam_img(img.copy())
            ret, img = self.cap2.read()
            self.new_cam_img2 = self.resize_cam_img2(img.copy())

            cv2.imshow("keypoint_img", self.tracking_img)
            cv2.imshow('img_bbx_from_detection', self.show_img_bbx_from_detection)

            k = cv2.waitKey(1)
            if k == ord('q'):
                self.keypoint_detected = 0
            if k == 27:
                self.running = 0
                self.keypoint_detected = 0

    def run_net(self):
        scale = []
        sqrt = math.sqrt(2)
        for i in range(5):
            scale.append(1 / math.pow(sqrt, i))
        # [1.0, 0.7071067811865475, 0.4999999999999999, 0.3535533905932737, 0.24999999999999994,
        #  0.1767766952966368, 0.12499999999999994]

        while self.running:
            '''
            Get a frame from camera and flip, crop
            '''
            # ret, img = self.cap.read()
            img = self.new_cam_img.copy()

            img_from_cam = img.copy()
            img = cv2.resize(img, (128, 128))
            self.show_img_bbx_from_detection = img

            # prepare the pyramid img tensor list
            img_list = helper.pyr_img_list(img, scale)
            temp_img_r = self.show_img_bbx_from_detection.copy()
            tensor_img = self.gen_tensor_imglist(img_list)

            # feed img_tensor_list to Pnet, and get bbx result
            cls_pred, box_offset_pred = self.get_bbx_cls_from_pnet(tensor_img)
            boxes = helper.result_process(scale, cls_pred, box_offset_pred, 0.5, nms_threshold=0.1)

            # select the most big face in the result
            box_list = [0, 0, 0, 0]
            w_max = -1
            if boxes is not None:
                # init threshold and w_max
                r_cls_threshold_init = self.cls_r_threshold
                w_max_th = 0

                for sbox in boxes:
                    x1, y1, x2, y2, w_b, h_b = self.get_bbx_info(sbox)
                    if w_b < 0 or h_b < 0:
                        continue
                    w_max, center = helper.get_center_w_from_two_points(x1, y1, x2, y2)
                    if w_max > w_max_th:
                        p1, p2 = helper.gen_square_from_center_size(w_max, center)
                        w_max_th = w_max
                        try:
                            img_r = self.crop_img_from_two_points(temp_img_r, p1, p2)

                            self.cls_r, offset_r = self.get_cls_from_rnet(img_r)
                            if self.cls_r > r_cls_threshold_init:
                                box_list = [x1, y1, w_b, h_b]
                                r_cls_threshold_init = self.cls_r
                            cv2.rectangle(self.show_img_bbx_from_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        except BaseException as err:
                            print('error: ', err)

                if self.cls_r > self.cls_r_threshold and sum(box_list) > 0:
                    box_list = np.array(box_list) / 128. * self.cam_size
                    w_max = w_max / 128. * self.cam_size

                    box_list = box_list.astype(np.int)

                    topleft = box_list[0:2]
                    x = int(topleft[0])
                    y = int(topleft[1])
                    w_max = int(w_max)

                    print(w_max, topleft)
                    keypoint_img = img_from_cam[y:y + w_max, x:x + w_max, :]

                    self.kal = KalmanFilter(1, np.array([w_max]).astype(np.float32))
                    # cv2.imshow("keypoint_img", keypoint_img)

                    self.keypoint_size = w_max
                    self.keypoint_topleft = [x, y]
                    self.run_keypoint(keypoint_img, img_from_cam, w_max, topleft)

            k = cv2.waitKey(1)
            if k == 27:
                self.running = 0
                break

    @staticmethod
    def crop_img_from_two_points(temp_img_r, p1, p2):
        img_r = temp_img_r[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0]), :]
        return img_r

    def detect_box(self, w_max_th, temp_img_r, x1, y1, x2, y2, w_b, h_b, r_cls_threshold_init, box_list):
        w_max, center = helper.get_center_w_from_two_points(x1, y1, x2, y2)
        if w_max > w_max_th:
            p1, p2 = helper.gen_square_from_center_size(w_max, center)
            w_max_th = w_max
            try:
                img_r = temp_img_r[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0]), :]

                self.cls_r, offset_r = self.get_cls_from_rnet(img_r)
                if self.cls_r > r_cls_threshold_init:
                    box_list = [x1, y1, w_b, h_b]
                    r_cls_threshold_init = self.cls_r
                cv2.rectangle(self.show_img_bbx_from_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except BaseException as err:
                print('error: ', err)
            return w_max_th, box_list, r_cls_threshold_init

    def run_keypoint(self, current_img, origin_img, size, topleft):
        # current_img = cv2.imread('saved_img.jpg')

        t1 = 0
        while self.running:
            try:
                if self.keypoint_detected == 1:
                    img = self.new_cam_img
                    origin_img = img.copy()

                    current_img = origin_img[self.keypoint_topleft[1]:(self.keypoint_topleft[1] + self.keypoint_size),
                                  self.keypoint_topleft[0]:(self.keypoint_topleft[0] + self.keypoint_size), :]

                    img_size = self.keypoint_size
                    topleft = self.keypoint_topleft
                else:
                    origin_img = origin_img.copy()
                    img_size = self.img_size
                    raw = current_img.copy()
                    img_size = size

                    rsize = 96
                    im_crop = cv2.resize(current_img.copy(), (rsize, rsize), interpolation=cv2.INTER_NEAREST)

                    im_crop = self.transform1(im_crop)
                    im_crop = im_crop[np.newaxis, :, :, :]

                    tensor_img = torch.cuda.FloatTensor(im_crop.cuda()) * 2 - 1  # GPU

                    point_list, cls = self.modelO(tensor_img)
                    cls = cls.cpu().data.numpy()[0][0]
                    point_list = point_list.cpu().data.numpy()[0]
                    point_list = point_list + self.point_off
                    self.kal_points = KalmanFilter(136, point_list.astype(np.float32))
                    self.kal_cls = KalmanFilter(1, np.array([cls]).astype(np.float32))
                    self.kal_distance = KalmanFilter(1, np.array([150]).astype(np.float32))

                    self.keypoint_detected = 1
                    continue

                rsize = 96
                im_crop = cv2.resize(current_img.copy(), (rsize, rsize), interpolation=cv2.INTER_NEAREST)

                im_crop = self.transform1(im_crop)
                im_crop = im_crop[np.newaxis, :, :, :]

                tensor_img = torch.cuda.FloatTensor(im_crop.cuda()) * 2 - 1  # GPU

                t1 = time.clock()
                point_list, cls = self.modelO(tensor_img)
                cls = cls.cpu().data.numpy()[0][0]

                print('FPS: ', 1 / (time.clock() - t1), 'Face: ', cls)
                # t1 = time.clock()

                point_list = point_list.cpu().data.numpy()[0]
                point_list = point_list + self.point_off

                point_list = self.kal_points(point_list.astype(np.float32))
                cls = self.kal_cls(np.array([cls]).astype(np.float32))

                points = []
                # print(topleft, img_size)
                for i in range(68):
                    x = int((point_list[i * 2]) * img_size) + topleft[0]
                    y = int((point_list[i * 2 + 1]) * img_size) + topleft[1]
                    points.append(tuple([x, y]))

                im_crop = origin_img

                self.keypoint_size, center, self.keypoint_topleft = self.get_bbx_from_point(points)

                # face outline
                for point in points[0:17]:
                    cv2.circle(im_crop, point, 0, (255, 255, 255), 5)

                # eyebrow
                for point in points[17:27]:
                    cv2.circle(im_crop, point, 0, (255, 120, 255), 7)

                # nose
                for point in points[27:31]:
                    cv2.circle(im_crop, point, 0, (255, 120, 0), 7)
                for point in points[31:36]:
                    cv2.circle(im_crop, point, 0, (120, 255, 0), 7)

                # ###################################################
                # # eyes
                # for point in points[36:42]:
                #     cv2.circle(im_crop, point, 0, (0, 120, 255), 8)
                # for point in points[42:48]:
                #     cv2.circle(im_crop, point, 0, (0, 255, 120), 8)
                #
                # # mouth
                # for point in points[49:68]:
                #     cv2.circle(im_crop, point, 0, (255, 0, 255), 5)
                # ###################################################

                # eyes
                # left
                cv2.circle(im_crop, points[36], 0, (0, 120, 255), 8)
                cv2.circle(im_crop, points[40], 0, (0, 120, 255), 8)
                # right
                cv2.circle(im_crop, points[42], 0, (0, 255, 120), 8)
                cv2.circle(im_crop, points[45], 0, (0, 255, 120), 8)

                # mouth
                cv2.circle(im_crop, points[48], 0, (255, 0, 255), 5)
                cv2.circle(im_crop, points[54], 0, (255, 0, 255), 5)
                ###################################################

                # draw rectangle on processed img
                downright = np.array(topleft) + img_size
                downright = np.int_(downright)

                x1, y1 = topleft
                x2, y2 = downright
                self.distance_img = im_crop[y1:y2, x1:x2, :]
                distance=0
                if self.distance_img is not None:
                    try:
                        distance = int(self.detect_distance())
                    except:
                        distance = 0
                # print(distance)
                cv2.rectangle(im_crop, tuple(topleft), tuple(downright), (255, 0, 0), 4)

                cv2.putText(im_crop, str(distance), tuple(topleft),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

                if cls < 0.1:
                    self.keypoint_detected = 0
                    self.tracking_img = im_crop
                    cv2.putText(self.tracking_img, 'Face Lost', tuple(topleft),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
                    break
                if self.keypoint_detected == 0:
                    self.tracking_img = im_crop
                    cv2.putText(self.tracking_img, 'Face Reset', tuple(topleft),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
                    break

                self.tracking_img = im_crop
                # cv2.imshow('keypoint', im_crop)
            except:
                self.keypoint_detected = 0
                break

    def detect_distance(self):
        right_img = self.new_cam_img2.copy()
        right_img = cv2.resize(right_img, (0, 0), fx=0.25, fy=0.25)
        match_img = self.distance_img.copy()
        match_img = cv2.resize(match_img, (0, 0), fx=0.25, fy=0.25)
        # w, h, ch = match_img.shape
        method = cv2.TM_CCOEFF
        res = cv2.matchTemplate(right_img, match_img, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        xr = self.keypoint_topleft[0] + 105
        xt = top_left[0]

        d = xr - xt
        f = 0.005
        b = 0.1
        distance = f * b / d * 1080 * 1000 * 100 / 2

        distance = self.kal_distance(np.array([distance]).astype(np.float32))
        if distance > 5000:
            distance = 5000

        return distance

    def convert_point_to_tuple(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p1[p1 < 0] = 0
        p1[p1 > self.cam_size] = self.cam_size
        p2[p2 < 0] = 0
        p2[p2 > self.cam_size] = self.cam_size
        p1 = tuple(p1)
        p2 = tuple(p2)
        return p1, p2

    def resize_cam_img(self, img):
        img = img.copy()
        img = cv2.flip(img, 1)
        img = img[0:, self.cam_left_size:self.cam_left_size + self.cam_size, :]
        return img

    def resize_cam_img2(self, img):
        img = img.copy()
        img = cv2.flip(img, 1)
        # img = img[0:, self.cam_left_size:self.cam_left_size + self.cam_size, :]
        return img

    def get_bbx_from_point(self, points):
        points = np.array(points)
        # center = np.array(points[7]).astype(np.uint8)
        x_list = np.array(points[:, 0])
        y_list = np.array(points[:, 1])

        max_x = np.max(x_list)
        min_x = np.min(x_list)
        w = max_x - min_x

        max_y = np.max(y_list)
        min_y = np.min(y_list)
        h = max_y - min_y

        size = max(w, h) * 1.4

        center = np.array([(max_x + min_x) / 2, (max_y + min_y - 0.0 * h) / 2])

        topleft = center - np.array([size / 2, size / 2])

        size = self.kal(np.array([size]).astype(np.float32))
        size = int(size)
        topleft = topleft.astype(np.int)

        topleft[topleft < 0] = 0
        if topleft[0] + size > self.cam_size:
            topleft[0] = self.cam_size - size
        if topleft[1] + size > self.cam_size:
            topleft[1] = self.cam_size - size

        # print(size, topleft)

        return size, center, topleft

    def get_bbx_cls_from_pnet(self, tensor_img):
        cls_pred = []
        box_offset_pred = []

        for _ in tensor_img:
            cls, box_offset = self.modelP(_)
            cls = cls.cpu().data.numpy()[0][0]
            box_offset = box_offset.cpu().data.numpy()[0].transpose((1, 2, 0))
            cls_pred.append(cls)
            box_offset_pred.append(box_offset)

        return cls_pred, box_offset_pred

    def get_cls_from_rnet(self, img):
        img = cv2.resize(img, (24, 24))
        tensor_img = helper.img_transform(img)
        cls_pred, box_offset_pred = self.modelR(tensor_img)

        result_r = cls_pred.cpu().data.numpy()[0][0]
        offset_r = box_offset_pred.cpu().data.numpy()[0]
        return result_r, offset_r

    @staticmethod
    def gen_tensor_imglist(img_list):
        tensor_img = []
        for img_ in img_list:
            tensor_img.append(helper.img_transform(img_))
        return tensor_img

    @staticmethod
    def get_bbx_info(sbox):
        box_np = sbox[0:4]
        box_np = box_np.astype(np.uint8)

        x1, y1, x2, y2 = box_np
        w_b = x2 - x1
        h_b = y2 - y1

        return x1, y1, x2, y2, w_b, h_b


class KalmanFilter:
    def __init__(self, num_filter_points, pre):
        self.kalman = cv2.KalmanFilter(num_filter_points, num_filter_points, 0)
        self.kalman.measurementMatrix = np.eye(num_filter_points, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(num_filter_points, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(num_filter_points, dtype=np.float32) * 1e-3
        self.kalman.measurementNoiseCov = np.eye(num_filter_points, dtype=np.float32) * 1e-3
        self.num_filter = num_filter_points
        self.kalman.statePre = pre
        self.kalman.statePost = pre

    def __call__(self, pts, *args, **kwargs):
        # pts_measure = pts[0][:, 0]
        pts_measure = np.reshape(pts, (self.num_filter, 1)).astype(np.float32)

        self.kalman.correct(pts_measure)

        return self.kalman.predict()


if __name__ == '__main__':
    bbx = FaceBBox()
