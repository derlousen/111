1. generate a random box 

2. check iou of this box

3. calculate the offset of this box

4. size related to net, p net input 12x12, all the regression coordinate to this size

negative sample use 0 offset for bounding box regression

https://zhuanlan.zhihu.com/p/31761796

https://github.com/dlunion/mtcnn
https://github.com/dlunion/mtcnn/tree/master/train
https://blog.csdn.net/AMDS123/article/details/69568495

https://github.com/dlunion/mtcnn/blob/master/train/gen_12net_data2.py#L113