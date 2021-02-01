import os
import tqdm
import pandas as pd
from PIL import Image, ImageDraw
from absl.flags import FLAGS
from absl import app, flags
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch.utils.data import DataLoader
from model.backbone import resnet
from model.rdd import RDD
from data.dataset import DOTA
from data.aug.compose import Compose
from data.aug import ops
from collections import defaultdict
from pathlib import Path
from utils.parallel import CustomDetDataParallel
from utils.box.bbox_np import xywha2xy4, xy42xywha
from utils.box.rbbox_np import rbbox_batched_nms
from multiprocessing import Pool
import math
import ntpath
from disjoint_set import DisjointSet
from utils.box.rbbox_np import rbbox_iou

#TODO: FLAGS
flags.DEFINE_string('input_folder', '../images/train', 'path to folder containing jpgs to predict')
flags.DEFINE_string('weights', '../save/weight/128700.pth', 'path to weights file')
flags.DEFINE_string('output_folder', '../predictions', 'folder where to save cropped jpgs')
flags.DEFINE_string('visualized_predicted_bbs_folder', '../predictions/bbs', '...')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('num_workers', 4, '')
flags.DEFINE_float('conf_thresh', 0.01, '...')
flags.DEFINE_float('nms_thresh', 0.01, '...')
flags.DEFINE_integer('image_size', 768, '')
flags.DEFINE_integer('w_bb_margin', 100, '')
flags.DEFINE_integer('h_bb_margin', 20, '')
flags.DEFINE_string('GPUS', '0', 'comma separated gpu ids')


def angle_diff(a1, a2):
    diff = a1 - a2
    diff = abs((diff + 180) % 360 - 180)
    return diff

def create_dataset_loader(input_folder):
    def create_dataset_from_folder(dir_dataset):
        pairs = []
        for filename in os.listdir(dir_dataset):
            img = os.path.join(dir_dataset, filename)
            anno = None
            pairs.append([img, anno])

        dataset_path = os.path.join(FLAGS.output_folder, 'image-sets', 'dataset.json')
        os.makedirs(Path(dataset_path).parent, exist_ok=True)
        json.dump(pairs, open(dataset_path, 'wt'), indent=2)

    create_dataset_from_folder(FLAGS.input_folder)
    torch.cuda.set_device(int(FLAGS.GPUS.split(',')[0]))

    aug = Compose([ops.PadSquare(), ops.Resize(FLAGS.image_size)])
    dataset = DOTA(FLAGS.output_folder, 'dataset', aug)


    loader = DataLoader(dataset, FLAGS.batch_size,
                        num_workers=FLAGS.num_workers,
                        pin_memory=True,
                        collate_fn=dataset.collate)
    print(f'created dataset from  {len(dataset)} files')
    return loader, len(dataset.names)

def load_model(num_classes):
    print('loading model...')

    dir_weight = os.path.join(FLAGS.weights) #later replace for FLAGS.weigths

    prior_box = {
        'strides': [8, 16, 32, 64, 128],
        'sizes': [3] * 5,
        'aspects': [[1, 2, 4, 8]] * 5,
        'scales': [[2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]] * 5,
        'old_version': False
    }
    cfg = {
        'prior_box': prior_box,
        'num_classes': num_classes,
        'extra': 2,
        'conf_thresh': FLAGS.conf_thresh,
        'nms_thresh': FLAGS.nms_thresh
    }

    backbone = resnet.resnet101
    model = RDD(backbone(fetch_feature=True), cfg)
    model.build_pipe(shape=[2, 3, FLAGS.image_size, FLAGS.image_size])
    model.restore(dir_weight)
    device_ids = [int(gpu_id) for gpu_id in FLAGS.GPUS.split(',')]
    if len(device_ids) > 1:
        model = CustomDetDataParallel(model, device_ids)
    model.cuda()
    model.eval()
    print('model restored')
    return model


def predict(model, data_loader, output_csv):
    print('predicting...')
    count_predicted, count_not_predicted = 0, 0
    ret_raw = defaultdict(list)

    with torch.no_grad():
        for images, targets, infos in tqdm.tqdm(data_loader):
            images = images.cuda() / 255
            dets = model(images)
            for (det, info) in zip(dets, infos):
                fname = ntpath.basename(info['img_path'])
                if det:
                    bboxes, scores, labels = det
                    bboxes = bboxes.cpu().numpy()
                    scores = scores.cpu().numpy()
                    labels = labels.cpu().numpy()
                    x, y, w, h = 0, 0, info['shape'][1], info['shape'][0]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    long_edge = max(w, h)
                    pad_x, pad_y = (long_edge - w) // 2, (long_edge - h) // 2
                    bboxes = np.stack([xywha2xy4(bbox) for bbox in bboxes])
                    bboxes *= long_edge / FLAGS.image_size
                    bboxes -= [pad_x, pad_y]
                    bboxes += [x, y]
                    bboxes = np.stack([xy42xywha(bbox, flag=1) for bbox in bboxes])

                    ret_raw[fname].append([bboxes, scores, labels])
                    count_predicted += 1
                else:
                    count_not_predicted += 1
                    ret_raw[fname].append([np.zeros((1, 5)), np.zeros((1,)), np.zeros((1,))])

    print(f'{count_predicted} jpgs with at least one object found, {count_not_predicted} with no object found')

    print('merging results...')
    ret = []
    for fname, dets in tqdm.tqdm(ret_raw.items()):
        bboxes, scores, labels = zip(*dets)
        bboxes = np.concatenate(list(bboxes))
        scores = np.concatenate(list(scores))
        labels = np.concatenate(list(labels))
        keeps = rbbox_batched_nms(bboxes, scores, labels, FLAGS.nms_thresh)
        ret.append([fname, [bboxes, scores[keeps], labels[keeps]]])

    columns = ['img_name', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'angle_class', 'angle_class_degrees',
               'angle_class_degrees_prob', 'cx', 'cy', 'w', 'h', 'bb_angle']
    df = pd.DataFrame(columns=columns)
    idx = 0
    print('exporting csv...')
    for fname, (bboxes, scores, labels) in tqdm.tqdm(ret):
        for bbox, score, label in zip(bboxes, scores, labels):
            angle_step = 5
            xy4 = xywha2xy4(bbox).ravel()
            row = [
                    fname,
                    xy4[0], xy4[1],
                    xy4[2], xy4[3],
                    xy4[4], xy4[5],
                    xy4[6], xy4[7],
                    label, label*angle_step, score,
                    bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                ]
            df.loc[idx] = row
            idx += 1

    df.to_csv(output_csv, sep=';')
    print('prediction results saved to', output_csv)


def filter_overlaping_bbs(input_csv, output_csv):
    df = pd.read_csv(input_csv, sep=';')
    df['keep'] = False

    iou_threshold = 0.8

    print('filtering overlaping bbs...')
    for img_name, group in tqdm.tqdm(df.groupby('img_name')):

        ds = DisjointSet()

        for start_idx, (idx1, row1) in enumerate(group.iterrows()):
            ds.union(idx1, idx1)

            xywha1 = np.array(list([row1['cx'], row1['cy'], row1['w'], row1['h'], row1['bb_angle']]))

            g2 = list(group.iterrows())
            for j in range(start_idx, len(g2)):

                idx2, row2 = g2[j]

                if idx1 != idx2:
                    # if bbs are identical iou is 0 insteed of 1, probably a bug, so a small number is added
                    e = 0.00001
                    xywha2 = np.array(list([row2['cx'], row2['cy'], row2['w'], row2['h'], row2['bb_angle'] + e]))
                    iou = rbbox_iou(xywha1, xywha2)
                    #if bbs overlaps, join them in one cluster,
                    if iou > iou_threshold:
                        ds.union(idx1, idx2)

        #select one final bb with largest score in every cluster
        for disjointset in ds.itersets():
            idxs = list(disjointset)
            g = group.loc[idxs]
            scores = g.angle_class_degrees_prob.values
            keepidx = idxs[np.argmax(scores)]
            df.loc[keepidx, 'keep'] = True
            group.loc[keepidx, 'keep'] = True

    df = df[df.keep==True]

    df.to_csv(output_csv,  sep=';')


def visualize_bbs(img_bbs):
    img_name = img_bbs[0]['img_name']
    img_path = os.path.join(FLAGS.input_folder, img_name)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for bb_row in img_bbs:
        draw.line((bb_row['x1'], bb_row['y1'], bb_row['x2'], bb_row['y2']), fill='red', width=10)
        draw.line((bb_row['x2'], bb_row['y2'], bb_row['x3'], bb_row['y3']), fill='red', width=10)
        draw.line((bb_row['x3'], bb_row['y3'], bb_row['x4'], bb_row['y4']), fill='red', width=10)
        draw.line((bb_row['x4'], bb_row['y4'], bb_row['x1'], bb_row['y1']), fill='red', width=10)
    img.save(os.path.join(FLAGS.visualized_predicted_bbs_folder, img_name))


def export_cropped_bb(img_bbs):
    img_name = img_bbs[0]['img_name']
    img_path = os.path.join(FLAGS.input_folder, img_name)
    img = Image.open(img_path)

    for i, bb_row in enumerate(img_bbs):

        cx, cy, w, h, bb_angle = bb_row['cx'], bb_row['cy'], bb_row['w'], bb_row['h'], bb_row['bb_angle']
        if w == 0 or h == 0:
            continue

        predicted_img_angle = bb_row['angle_class_degrees']

        bb_angle_from_counter_clockwise = 360 - bb_angle
        if angle_diff(bb_angle_from_counter_clockwise, predicted_img_angle) > 90:
            bb_angle = (bb_angle + 180) % 360

        rotated_img = img.rotate(bb_angle, center=(cx, cy))
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        x_min, y_min = min(x1,x2), min(y1,y2)
        x_max, y_max = max(x1,x2), max(y1,y2)

        w_margin, h_margin = FLAGS.w_bb_margin, FLAGS.h_bb_margin

        if x_max - x_min < y_max - y_min:
            w_margin, h_margin = h_margin, w_margin
        x_max += w_margin
        x_min -= w_margin
        y_max += h_margin
        y_min -= h_margin

        cropped_bar = rotated_img.crop((x_min, y_min, x_max, y_max))
        p = os.path.join(FLAGS.output_folder, 'cropped_bbs')
        img_name_without_extension = '.'.join(img_name.split('.')[:-1])
        img_extension = img_name.split('.')[-1]
        cropped_bar.save(os.path.join(p, f'{img_name_without_extension}__{i}.'+img_extension))


def crop_images(prediction_output_csv):

    df = pd.read_csv(prediction_output_csv, sep=';')

    image_predictions = dict()

    for _, row in df.iterrows():
        img_name = row['img_name']
        if img_name not in image_predictions:
            image_predictions[img_name] = []
        image_predictions[img_name].append(row)

    converted_dict_to_list = []
    for img_name in image_predictions:
        converted_dict_to_list.append(image_predictions[img_name])

    if FLAGS.visualized_predicted_bbs_folder is not None:
        os.makedirs(FLAGS.visualized_predicted_bbs_folder, exist_ok=True)
        print('saving imgs with drawn bbs to', FLAGS.visualized_predicted_bbs_folder)
        with Pool(processes=FLAGS.num_workers) as pool:
            pool.map(visualize_bbs, converted_dict_to_list)

    p = os.path.join(FLAGS.output_folder, 'cropped_bbs')
    print('Exporting cropped bbs to', p)
    os.makedirs(p, exist_ok=True)
    with Pool(processes=FLAGS.num_workers) as pool:
        pool.map(export_cropped_bb, converted_dict_to_list)
    print('all done.')

def main(argv):


    data_loader, num_classes = create_dataset_loader(FLAGS.input_folder)
    model = load_model(num_classes = num_classes) #same as 360 // 5(angle_step)

    predictions_all_bbs = os.path.join(FLAGS.output_folder, 'predictions_all_bbs.csv')

    predict(model, data_loader, predictions_all_bbs)

    predictions_final_bbs = os.path.join(FLAGS.output_folder, 'predictions_final_bbs.csv')

    filter_overlaping_bbs(predictions_all_bbs, predictions_final_bbs)

    crop_images(predictions_final_bbs)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass