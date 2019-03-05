import cv2
import numpy as np

img_name = 'test2.PNG'

cv2.namedWindow('img', 0)
img = cv2.imread(img_name)
imCrop = cv2.resize(img, (48, 48), interpolation=cv2.INTER_NEAREST)

point_off = np.load('offset.np')


def rotate(input_img, points, angle=0):
    # anti clock angle

    rows, cols = input_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    res = cv2.warpAffine(input_img, M, (rows, cols))
    angle_M = M[:, :2]
    offset = M[:, 2:].transpose()[0] / 48
    point_off = np.matmul(angle_M, points.transpose()).transpose()
    point_off = point_off + offset
    return res, point_off


def flip_img(input_img, points):
    input_img = cv2.flip(input_img, 1)
    points[:, 0] = 1 - points[:, 0]
    return input_img, points


# rows,cols = img.shape[:2]
# M = cv2.getRotationMatrix2D((cols/2,rows/2), 45, 1)
# res = cv2.warpAffine(imCrop, M, (48, 48))
# angle_M = M[:, :2]
# offset = M[:, 2:].transpose()[0]/48
# point_off = np.matmul(angle_M, point_off.transpose()).transpose()
# point_off = point_off+offset


res, point_off = rotate(imCrop, point_off, angle=-45)
res, point_off = flip_img(res, point_off)

point_off = point_off.flatten()

off_circles = []

for i in range(68):
    x = int((point_off[i * 2]) * 48)
    y = int((point_off[i * 2 + 1]) * 48)
    off_circles.append(tuple([x, y]))

for off_circle in off_circles:
    cv2.circle(res, off_circle, 0, (0, 255, 0), 1)

cv2.imshow('img', res)
k = cv2.waitKey(0)
