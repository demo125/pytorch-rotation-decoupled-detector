import lxml.etree
import glob
import os
import math
from PIL import Image, ImageDraw
import numpy as np
import ntpath
from shutil import copyfile


def get_xml_data(xml_path):
    def parse_xml(xml):
        if not len(xml):
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = parse_xml(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    annotation_xml = lxml.etree.fromstring(open(xml_path).read())
    annotation = parse_xml(annotation_xml)['annotation']
    boxes = []
    for obj in annotation['object']:
        box = {k: float(i) for k, i in obj['robndbox'].items()}
        class_name = obj['name']
        size = annotation['size']
        angle = math.degrees(float(box['angle']))

        box['img_width'] = float(size['width'])
        box['img_height'] = float(size['height'])
        box['angle'] = angle
        box['class'] = class_name
        boxes.append(box)
    return boxes

def rotate_around_point(origin, radians, point):
    """Rotate a point around a given point.

    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy

def get_points(box):

    cx, cy = box['cx'], box['cy']
    w2, h2 = box['w']/2, box['h']/2
    xlt = cx - w2; ylt = cy - h2
    xrt = cx + w2; yrt = cy - h2

    xlb = cx - w2; ylb = cy + h2
    xrb = cx + w2; yrb = cy + h2

    xlt, ylt = rotate_around_point((cx, cy), math.radians(box['angle']), (xlt, ylt))
    xrt, yrt = rotate_around_point((cx, cy), math.radians(box['angle']), (xrt, yrt))
    xlb, ylb = rotate_around_point((cx, cy), math.radians(box['angle']), (xlb, ylb))
    xrb, yrb = rotate_around_point((cx, cy), math.radians(box['angle']), (xrb, yrb))
    return (xlt, ylt, xrt, yrt, xrb, yrb, xlb, ylb)



xml_files = list(glob.iglob(os.path.join('/home/mde/python/veolia-amr/data/annotated/all_annotated', '*.xml')))

for d in ['train', 'val', 'test']:
    for d2 in ['images', 'labelTxt']:
        p = os.path.join(d2, d)
        if not os.path.isdir(p):
            os.makedirs(p)

d = {
    'train': 0,
    'test': 0,
    'val': 0
}
for i, xml_file in enumerate(xml_files):
    jpg_file = xml_file[:-4] + '.jpg'
    xml_filename = ntpath.basename(xml_file)
    jpg_filename = xml_filename[:-4] + '.jpg'
    txt_filename = xml_filename[:-4] + '.txt'

    boxes = get_xml_data(xml_file)
    labels = []
    for box in boxes:
        points = get_points(box)

        if box['class'] == 'register':
            if box['angle'] < 180:
                box['class'] = 'register_0-179'
            else:
                box['class'] = 'register_180-360'

        # xywha = box['cx'], box['cy'], box['w'], box['h'], 360 - box['angle']
        # xywha = (float(i) for i in xywha)
        # xy4 = xywha2xy4(xywha)
        # back_xywha = xy42xywha(xy4)

        label = ' '.join([str(x) for x in points]) + f" {box['class']} {box['angle']}"
        labels.append(label)

    dest = None
    if np.random.random() > 0.0:
        if np.random.random() > 0.15:
            dest = 'train'
        else:
            dest = 'val'
    else:
        dest = 'test'

    d[dest] += 1

    o_dir = os.path.join('./labelTxt', dest)
    os.makedirs(o_dir, exist_ok=True)
    p = os.path.join(o_dir, txt_filename)
    with open(p, 'w+') as f:
        f.write('\n'.join(labels))
        f.close()

    o_dir = os.path.join('images', dest)
    os.makedirs(o_dir, exist_ok=True)

    new_img_path = os.path.join(o_dir, jpg_filename)
    if not os.path.isfile(new_img_path):
        copyfile(jpg_file, new_img_path)

print(d)
