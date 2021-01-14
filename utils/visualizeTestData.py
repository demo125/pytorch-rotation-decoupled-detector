from PIL import Image, ImageDraw
import os

with open('../save/submission/Task1_bar.txt') as f:
    lines = f.readlines()
    for i, l in enumerate(lines):
        parts = l.replace('\n','').split(' ')
        jpg_name = parts[0]
        print(i, len(lines), jpg_name)
        img = Image.open(os.path.join('../images/test', jpg_name))
        draw = ImageDraw.Draw(img)
        coords = tuple(float(x) for x in parts[2:])
        # draw.polygon(coords, outline=(0,0,255), fill=None)

        draw.line(coords, fill="red", width=9)
        draw.line((coords[-2], coords[-1], coords[0], coords[1]), fill="red", width=9)

        img.save(os.path.join('predictions-second', str(i)+'_'+jpg_name))
