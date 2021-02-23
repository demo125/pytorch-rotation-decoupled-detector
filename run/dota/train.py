# -*- coding: utf-8 -*-
# File   : train.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 11:10
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import sys
#import pyximport
#pyximport.install()
import numpy as np
from PIL import Image
sys.path.append('.')

import os
import tqdm
import torch

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.aug.compose import Compose
from data.aug import ops
from data.dataset import DOTA

from model.rdd import RDD
from model.backbone import resnet

from utils.adjust_lr import adjust_lr_multi_step
from utils.parallel import convert_model, CustomDetDataParallel
from utils.box.bbox_np import xy42xywha, xywha2xy4
import matplotlib.pyplot as plt

from absl.flags import FLAGS
from absl import app, flags

flags.DEFINE_string('train_json_path', '/home/mde/python/pytorch-rotation-decoupled-detector/image-sets/train.json',
                    'dir with train images')
flags.DEFINE_string('val_json_path', '/home/mde/python/pytorch-rotation-decoupled-detector/image-sets/val.json',
                    'dir with val images')
flags.DEFINE_string('test_json_path', '/home/mde/python/pytorch-rotation-decoupled-detector/image-sets/test.json',
                    'dir with test images')
flags.DEFINE_string('dir_weight', '/home/mde/python/pytorch-rotation-decoupled-detector/save/weight',
                    'dir for saving weights')
flags.DEFINE_string('dir_logs', '/home/mde/python/pytorch-rotation-decoupled-detector/save/logs',
                    'dir for saving logs')

flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('restore_epoch', -1, 'restore epoch checkpoint, if -1, restore last epoch,'
                                          ' if 0 start new training')
flags.DEFINE_integer('num_workers', 50, 'num cores')
flags.DEFINE_integer('max_epochs', 300, 'max num epochs')
flags.DEFINE_integer('save_interval', 5, 'save weight after this number of epochs')
flags.DEFINE_integer('image_size', 768, 'model input image size')
flags.DEFINE_string('device_ids', '0,1,2,3', 'comma separated gpu ids, eg. "0,1,2"')

def restore_dataset(dataset_dirs, augment):
    if augment:
        aug = Compose([
            ops.ToFloat(),
            # ops.PhotometricDistort(),
            ops.RandomGray(),
            ops.RandomBrightness(20),
            ops.RandomContrast(0.9, 1.1),
            # ops.RandomLightingNoise(),
            ops.RandomRotate90(),
            ops.PadSquare(),
            ops.Resize(FLAGS.image_size),
            # ops.BBoxFilter(24 * 24 * 0.4)
        ])
    else:
        aug = Compose([
            ops.ToFloat(),
            ops.PadSquare(),
            ops.Resize(FLAGS.image_size),
        ])

    dataset = DOTA(dataset_dirs, aug)
    loader = DataLoader(dataset, FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers,
                        pin_memory=True, drop_last=True, collate_fn=dataset.collate)
    return loader

def restore_model(dir_weight, num_classes, restore_epoch=None):

    os.makedirs(dir_weight, exist_ok=True)

    backbone = resnet.resnet101

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
    }

    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
    }

    model = RDD(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, FLAGS.image_size, FLAGS.image_size])

    if restore_epoch == -1:
        indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
        restore_epoch = max(indexes) if indexes else 0
        FLAGS.restore_epoch = restore_epoch

    if restore_epoch:
        model.restore(os.path.join(dir_weight, '%d.pth' % restore_epoch))
    else:
        model.init()

    device_ids = [int(id) for id in FLAGS.device_ids.split(',')]

    if len(device_ids) > 1:
        model = convert_model(model)
        model = CustomDetDataParallel(model, device_ids)
    model.cuda()
    print('model restored on step', restore_epoch)
    return model

def main(args):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device_ids = [int(id) for id in FLAGS.device_ids.split(',')]
    torch.cuda.set_device(device_ids[0])

    train_loader = restore_dataset([FLAGS.train_json_path],
                                   augment=True)
    val_loader = restore_dataset([FLAGS.val_json_path], augment=False)

    print('# of train images', len(train_loader.dataset))
    print('# of val images', len(val_loader.dataset))

    model = restore_model(FLAGS.dir_weight, num_classes=len(train_loader.dataset.names),
                          restore_epoch=FLAGS.restore_epoch)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    writer = SummaryWriter(FLAGS.dir_logs)

    current_epoch = FLAGS.restore_epoch

    while current_epoch < FLAGS.max_epochs:

        current_epoch += 1
        print(f"\nepoch {current_epoch}/{FLAGS.max_epochs}")
        epoch_losses = {}
        for images, targets, infos in tqdm.tqdm(train_loader, ncols=100):
            # adjust_lr_multi_step(optimizer, current_step, lr_cfg, warm_up)

            images = images.cuda() / 255
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for key, val in list(losses.items()):
                #losses[key] = val.item()# what is this ?
                key = f'train_{key}'
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(val.item())

        with torch.no_grad():
            for images, targets, infos in val_loader:
                images = images.cuda() / 255
                losses = model(images, targets)
                for key, val in list(losses.items()):
                    key = f'val_{key}'
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(val.item())

        for key, val in list(epoch_losses.items()):
            mean_val = np.mean(val)
            print(f'{key}={np.round(mean_val, 3)}', end=' ')

            writer.add_scalar(key, mean_val, global_step=current_epoch)
        writer.flush()

        if current_epoch % FLAGS.save_interval == 0:
            print(f'\nsaving {current_epoch}. epoch')
            save_path = os.path.join(FLAGS.dir_weight, '%d.pth' % current_epoch)
            state_dict = model.state_dict() if len(device_ids) == 1 else model.module.state_dict()
            torch.save(state_dict, save_path)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
