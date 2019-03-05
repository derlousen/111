import cv2
import numpy as np
import torch
from torch.autograd import Variable
import math
from torchvision import transforms
import numpy as np


def pyr_img(img):
    raw_img = img.copy()

    img = cv2.resize(img, (128, 128))
    out_img = np.zeros((192, 128, 3), dtype=np.uint8)
    out_img[0:128, 0:128, :] = img
    out_img[128:192, 0:64, :] = cv2.resize(img, (64, 64))
    out_img[128:160, 64:96, :] = cv2.resize(img, (32, 32))
    out_img[128:144, 96:112, :] = cv2.resize(img, (16, 16))

    return out_img


def pyr_img_list(img, scale):
    out_img = []
    for _ in scale:
        out_img.append(cv2.resize(img, (0, 0), fx=_, fy=_))
    return np.array(out_img)


def result_process(scale_list, cls_pred, box_offset_pred, threshold, nms_threshold=0.1):
    # probs = cls_pred.cpu().data.numpy()[0][0]
    # offsets = box_offset_pred.cpu().data.numpy()[0].transpose((1, 2, 0))
    # offsets = box_offset_pred.cpu().data.numpy()[0]
    # print(offsets.shape)

    boxes = []

    for i in range(len(scale_list)):
        prob_1 = cls_pred[i]
        offset_1 = box_offset_pred[i]
        scale = scale_list[i]
        if len(_generate_bboxes(prob_1, offset_1, scale, threshold)) > 0:
            boxes.append(_generate_bboxes(prob_1, offset_1, scale, threshold)[0])

    # prob_1 = probs[0:59, 0:59]
    # offset_1 = offsets[0:59, 0:59, :]
    # scale = 1
    # if len(_generate_bboxes(prob_1, offset_1, scale, threshold)) > 0:
    #     boxes.append(_generate_bboxes(prob_1, offset_1, scale, threshold)[0])
    #
    # prob_2 = probs[64:91, 0:27]
    # offset_2 = offsets[64:91, 0:27, :]
    # scale = 0.5
    # if len(_generate_bboxes(prob_2, offset_2, scale, threshold)) > 0:
    #     boxes.append(_generate_bboxes(prob_2, offset_2, scale, threshold)[0])
    #
    # prob_3 = probs[64:73, 32:41]
    # offset_3 = offsets[64:73, 32:41, :]
    # scale = 0.25
    # if len(_generate_bboxes(prob_3, offset_3, scale, threshold)) > 0:
    #     boxes.append(_generate_bboxes(prob_3, offset_3, scale, threshold)[0])
    #
    # prob_4 = probs[64:67, 48:51]
    # offset_4 = offsets[64:67, 48:51, :]
    # scale = 0.125
    # if len(_generate_bboxes(prob_4, offset_4, scale, threshold)) > 0:
    #     boxes.append(_generate_bboxes(prob_4, offset_4, scale, threshold)[0])

    boxes = np.array(boxes)
    keep = nms(boxes, overlap_threshold=nms_threshold)
    # org_x = 128
    # result_x = (org_x-12)/2 +1

    if len(boxes) == 0:
        return None

    # keep = nms(boxes[:, 0:5], overlap_threshold=threshold)
    return boxes[keep]


# def run_first_stage(image, net, scale, threshold):
#     """Run P-Net, generate bounding boxes, and do NMS.
#     Arguments:
#         image: an instance of PIL.Image.
#         net: an instance of pytorch's nn.Module, P-Net.
#         scale: a float number,
#             scale width and height of the image by this number.
#         threshold: a float number,
#             threshold on the probability of a face when generating
#             bounding boxes from predictions of the net.
#     Returns:
#         a float numpy array of shape [n_boxes, 9],
#             bounding boxes with scores and offsets (4 + 1 + 4).
#     """
#
#     # scale the image and convert it to a float array
#     width, height, ch = image.shape
#     sw, sh = math.ceil(width * scale), math.ceil(height * scale)
#     # print(sw, sh)
#     # sw, sh = int(sw), int(sh)
#     img = cv2.resize(image, (sw, sh))
#
#     # cv2.imshow('result', image)
#     # k = cv2.waitKey(0)
#
#     img = np.asarray(img, 'float32')
#
#     img = Variable(torch.cuda.FloatTensor(_preprocess(img)), volatile=True)
#     output = net(img)
#
#     probs = output[0].cpu().data.numpy()[0, 1, :, :]
#     offsets = output[1].cpu().data.numpy()
#
#     # probs: probability of a face at each sliding window
#     # offsets: transformations to true bounding boxes
#     # print(probs  )
#     # cv2.imshow('result', image)
#     # k = cv2.waitKey(0)
#
#     boxes = _generate_bboxes(probs, offsets, scale, threshold)
#     if len(boxes) == 0:
#         return None
#
#     keep = nms(boxes, overlap_threshold=0.5)
#     result = boxes[keep]
#
#     # print(result)
#
#     return result


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.
    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.
    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    # print(np.max(probs), np.min(probs))
    inds = np.where(probs > threshold)

    # print('-----------------',inds)

    if inds[0].size == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, tx2, ty1, ty2 = [offsets[inds[0], inds[1], i] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # print(offsets)

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back

    # bounding_boxes = np.vstack([
    #     np.round((stride * inds[1] + 1.0) / scale * (1 + tx1)),
    #     np.round((stride * inds[0] + 1.0) / scale * (1 + ty1)),
    #     np.round((stride * inds[1] + 1.0 + cell_size) / scale * (1 + tx2)),
    #     np.round((stride * inds[0] + 1.0 + cell_size) / scale * (1 + ty2)),
    #     score,
    #     # offsets
    # ])


    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale + (cell_size / scale * tx1)),
        np.round((stride * inds[0] + 1.0) / scale + (cell_size / scale * ty1)),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale + (cell_size / scale * tx2)),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale + (cell_size / scale * ty2)),
        score,
        # offsets
    ])



    # print(bounding_boxes)

    return bounding_boxes.T


def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.
    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.
    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes

    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    # print(x1, y1, x2, y2, ids)

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].
    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    """Cut out boxes from the image.
    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.
    Returns:
        a float numpy array of shape [n, 3, size, size].
    """

    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')

        img_array = np.asarray(img, 'uint8')
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        # img_box = Image.fromarray(img_box)
        # img_box = img_box.resize((size, size), Image.BILINEAR)

        img_box = cv2.resize(img_box, (size, size))
        img_box = np.asarray(img_box, 'float32')

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes


def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.
    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.
    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.
        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list


def _preprocess(img):
    """Preprocessing step before feeding the network.
    Arguments:
        img: a float numpy array of shape [h, w, c].
    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img


def get_center_w_from_two_points(x1, y1, x2, y2):
    w_max = float(max((x2 - x1), (y2 - y1)))
    center = [float((x1 + x2) / 2), float((y1 + y2) / 2)]
    return w_max, center


def get_center_w_from_w_h(bbox):
    w_max = max(bbox[2], bbox[3])
    center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2*1.3]
    # center = [bbox[0], bbox[1]]
    return w_max, center


def gen_square_from_center_size(w_max, center):
    p1 = (int(center[0] - w_max / 2), int(center[1] - w_max / 2))
    p2 = (int(center[0] + w_max / 2), int(center[1] + w_max / 2))
    return p1, p2


transform1 = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def img_transform(img):
    img = transform1(img)
    img = img[np.newaxis, :, :, :]

    tensor_img = torch.cuda.FloatTensor(img.cuda()) * 2 - 1  # GPU
    return tensor_img


def gen_bbox_wh_from_wh_offset(bbox, offset_r):
    # offset_r=[x1,x2,y1,y2]

    x1_off = bbox[2] * offset_r[0]
    y1_off = bbox[3] * offset_r[2]

    x2_off = bbox[2] * offset_r[1]
    y2_off = bbox[3] * offset_r[3]

    x1 = bbox[0] + x1_off
    y1 = bbox[1] + y1_off
    w = bbox[2] + x2_off - x1_off
    h = bbox[3] + y2_off - y1_off
    return [x1, y1, w, h]


def start_control_bar(value1, value2):
    def high(x):
        # global value1
        value1[0] = x

    def low(x):
        # global value2
        value2[0] = x

    # img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image', 0)
    # cv2.setWindowProperty('image')
    # cv2.imshow('image', img)
    # create trackbars for color change
    cv2.createTrackbar('H', 'image', 0, 255, high)
    cv2.setTrackbarPos('H', 'image', 30)
    cv2.createTrackbar('L', 'image', 0, 255, low)
    cv2.setTrackbarPos('L', 'image', 20)
    cv2.waitKey(1)


if __name__ == '__main__':
    a = [0]
    b = [0]
    start_control_bar(a, b)
    import time

    img = np.zeros((300, 512, 3), np.uint8)
    while True:
        print(a, b)
        time.sleep(0.3)
        cv2.imshow('a', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
