import sys

sys.path.append('.')

import os
import json
import cv2 as cv
import numpy as np
import math
from utils.crop_image import Cropper
from absl.flags import FLAGS
from absl import app, flags

flags.DEFINE_float('test_split', 0.1, 'percentage of images used for testing')
flags.DEFINE_float('val_split', 0.1, 'percentage of training data used for validation')


def load_dota_annotation(dota_file_path):
    for file in os.listdir(dota_file_path):
        objs = []
        for i, line in enumerate(open(os.path.join(dota_file_path, file)).readlines()):
            line = line.strip()
            line_split = line.split(' ')
            if len(line_split) == 10:
                obj = dict()
                coord = np.array(line_split[:8], dtype=np.float32).reshape([4, 2])
                bbox = cv.boxPoints(cv.minAreaRect(coord)).astype(np.int).tolist()
                obj['name'] = int(float(line_split[9])) #line_split[8].lower()
                obj['bbox'] = bbox
                obj['angle'] = float(line_split[8])
                objs.append(obj)
            else:
                print('<skip line> %s' % line)
        if objs:
            json.dump(objs, open(os.path.join(dir_json, file.replace('txt', 'json')), 'wt'), indent=2)

def split_images(test_split, val_split):


def main():

    if image_set != 'test':
        dir_txt = os.path.join(dir_dataset, 'labelTxt', image_set)
        out_dir_json = os.path.join(dir_dataset, 'annotations', image_set)
        txt2json(dir_txt, out_dir_json)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass