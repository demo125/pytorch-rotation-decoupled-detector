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
    box = annotation['object'][0]['robndbox']
    size = annotation['size']
    box['img_width'] = size['width']
    box['img_height'] = size['height']
    angle = math.degrees(float(box['angle']))
    box['angle'] = angle
    return box

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

def get_points_and_angle(box):

    cx, cy = float(box['cx']), float(box['cy'])
    w2, h2 = float(box['w'])/2, float(box['h'])/2
    xlt = cx - w2; ylt = cy - h2
    xrt = cx + w2; yrt = cy - h2

    xlb = cx - w2; ylb = cy + h2
    xrb = cx + w2; yrb = cy + h2

    xlt, ylt = rotate_around_point((cx, cy), math.radians(box['angle']), (xlt, ylt))
    xrt, yrt = rotate_around_point((cx, cy), math.radians(box['angle']), (xrt, yrt))
    xlb, ylb = rotate_around_point((cx, cy), math.radians(box['angle']), (xlb, ylb))
    xrb, yrb = rotate_around_point((cx, cy), math.radians(box['angle']), (xrb, yrb))
    return (xlt, ylt, xrt, yrt, xrb, yrb, xlb, ylb, ), box['angle']



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
    jpg_file = xml_file[:-4]+ '.jpg'
    xml_filename = ntpath.basename(xml_file)
    jpg_filename = xml_filename[:-4] + '.jpg'
    txt_filename = xml_filename[:-4] + '.txt'

    box = get_xml_data(xml_file)
    points, angle = get_points_and_angle(box)
    a = str(angle // 5)
    label = ' '.join([str(x) for x in points]) + f' {str(angle)} {a}'

    #create new
    # dest = None
    # if np.random.random() > 0.1:
    #     dest = 'train'
    # else:
    #     dest = 'val' if np.random.random() > 0.5 else 'test'

    #add angle

    for dest in ['train', 'test', 'val']:
        p = os.path.join('../labelTxt-first', dest, txt_filename)
        if os.path.isfile(p):
            d[dest] += 1
            print(p)
            # with open(p, 'w+') as f:
            #     f.write(label)
            #     f.close()

    # with open(p, 'w+') as f:
    #     f.write(label)
    #     f.close()
    o_dir = os.path.join('images', dest)
    
    if not os.path.isdir(o_dir):
        os.makedirs(o_dir)

    new_img_path = os.path.join(o_dir, jpg_filename)
    if not os.path.isfile(new_img_path):
        # copyfile(jpg_file, new_img_path)
        pass

print(d)
