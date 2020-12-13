# -*- coding: utf-8 -*-
# File   : evaluate.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:10
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import sys

sys.path.append('.')

import os
import tqdm
import torch

from torch.utils.data import DataLoader

from data.aug import ops
from data.dataset import HRSC2016

from model.rdd import RDD
from model.backbone import resnet

from utils.box.metric import get_det_aps
from utils.parallel import CustomDetDataParallel


@torch.no_grad()
def main():
    global checkpoint
    if checkpoint is None:
        dir_weight = os.path.join(dir_save, 'weight')
        indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
        current_step = max(indexes)
        checkpoint = os.path.join(dir_weight, '%d.pth' % current_step)

    image_size = 768
    batch_size = 16
    num_workers = 4

    aug = ops.Resize(image_size)
    dataset = HRSC2016(dir_dataset, 'test', aug)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate)
    num_classes = len(dataset.names)

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1.5, 3, 5, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }
    conf_thresh = 0.01
    nms_thresh = 0.5
    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
        'conf_thresh': conf_thresh,
        'nms_thresh': nms_thresh
    }

    model = RDD(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, image_size, image_size])
    model.restore(checkpoint)
    if len(device_ids) > 1:
        model = CustomDetDataParallel(model, device_ids)
    model.cuda()
    model.eval()

    count = 0
    gt_list, det_list = [], []
    for images, targets, infos in tqdm.tqdm(loader):
        images = images.cuda() / 255
        dets = model(images)
        for target, det in zip(targets, dets):
            if target:
                bboxes = target['bboxes'].numpy()
                labels = target['labels'].numpy()
                gt_list.extend([count, bbox, 1, label] for bbox, label in zip(bboxes, labels))
            if det:
                bboxes, scores, labels = list(map(lambda x: x.cpu().numpy(), det))
                det_list.extend([count, bbox, score, label] for bbox, score, label in zip(bboxes, scores, labels))
            count += 1
    APs = get_det_aps(det_list, gt_list, num_classes)
    mAP = sum(APs) / len(APs)
    print('AP')
    for label in range(num_classes):
        print(f'{dataset.label2name[label]}: {APs[label]}')
    print(f'mAP: {mAP}')


if __name__ == '__main__':

    device_ids = [0, 1]
    torch.cuda.set_device(device_ids[0])
    backbone = resnet.resnet101

    dir_dataset = '<replace with your local path>'
    dir_save = '<replace with your local path>'
    checkpoint = None

    main()
