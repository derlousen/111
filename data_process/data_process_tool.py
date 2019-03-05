import numpy as np
import cv2
import numpy.random as npr
from time import clock


def get_bbox(img_in, label_in):
    height = img_in.shape[0]
    width = img_in.shape[1]

    # white = (255, 0, 0)
    raw_img = img_in.copy()
    point_list = np.array(label_in)

    points = point_list[[27, 33]]

    # for point in points:
    #     point = point.astype(np.int)
    #     cv2.circle(img_in, tuple(point), 1, white, 5)

    distance = abs(points[0][1] - points[1][1]).astype(np.int)
    face_point = np.array([points[1], points[1]]).astype(np.int)

    face_point[0][0] -= 1.6 * distance
    if face_point[0][0] < 0:
        face_point[0][0] = 0

    face_point[1][0] += 1.6 * distance
    if face_point[0][0] > width:
        face_point[0][0] = width - 1

    left_top = face_point[0]
    left_top[1] -= 1.6 * distance

    right_down = face_point[1]
    right_down[1] += 1.5 * distance

    # cv2.circle(img_in, tuple(face_point[0]), 1, white, 5)
    # cv2.circle(img_in, tuple(face_point[1]), 1, white, 5)
    #
    # cv2.rectangle(img_in, tuple(left_top), tuple(right_down), (0, 255, 0), 3)

    return raw_img, [left_top[0], left_top[1], right_down[0], right_down[1]]


def get_bbox_n(img, annotation):
    height, width, channel = img.shape
    size = npr.randint(40, min(width, height) / 2)
    # top_left
    nx = npr.randint(0, width - size)
    ny = npr.randint(0, height - size)
    # random crop
    crop_box = np.array([nx, ny, nx + size, ny + size])

    annotation = np.array(annotation)
    annotation = annotation[np.newaxis, :]

    iou = cal_iou(crop_box, annotation)

    cropped_im = img[ny: ny + size, nx: nx + size, :]
    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

    return resized_im, iou, crop_box


def cal_iou(box, boxes):
    """
    :param box: selected bounding box are
    :param boxes: ground truth
    :return:
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def get_bbox_offset(img, annotation):
    height, width, channel = img.shape

    x1, y1, x2, y2 = annotation
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

    delta_x = npr.randint(int(-w * 0.2), int(w * 0.2))
    delta_y = npr.randint(int(-h * 0.2), int(h * 0.2))

    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
    nx2 = nx1 + size
    ny2 = ny1 + size

    crop_box = np.array([nx1, ny1, nx2, ny2])
    offset_x1 = (x1 - nx1) / float(size)
    offset_y1 = (y1 - ny1) / float(size)
    offset_x2 = (x2 - nx2) / float(size)
    offset_y2 = (y2 - ny2) / float(size)

    annotation = np.array(annotation)
    annotation = annotation[np.newaxis, :]

    iou = cal_iou([nx1, ny1, nx2, ny2], annotation)

    # print(iou)

    cropped_im = img[ny1: ny2, nx1: nx2, :]

    cv2.imshow("img", cropped_im)
    cv2.waitKey(1)

    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

    return resized_im, iou, [offset_x1, offset_x2, offset_y1, offset_y2]


def nms(boxes, overlap_threshold=0.5, mode='union'):
    """ Pure Python NMS baseline. """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if mode is 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]

    return keep


def rotate(img, points, angle=0, offset=None, bboxsize=96):
    # anti clock angle

    rows, cols = img.shape[:2]
    x = 0
    y = 0
    if offset is not None:
        x1, x2, y1, y2 = offset
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

    M = cv2.getRotationMatrix2D((cols / 2 + x, rows / 2 + y), angle, 1)
    res = cv2.warpAffine(img, M, (rows, cols))
    angle_M = M[:, :2]
    offset = M[:, 2:].transpose()[0] / bboxsize
    point_off = np.matmul(angle_M, points.transpose()).transpose()
    point_off = point_off + offset
    return res, point_off


def flip_img(img, points):
    img = cv2.flip(img, 1)
    points[:, 0] = 1 - points[:, 0]
    return img, points


def flip_img_wd(img, offset):
    img = cv2.flip(img, 1)
    # print(offset)
    offset[0:2] = - offset[0:2]
    offset[0:2] = [offset[1], offset[0]]
    # print(offset)
    return img, offset
