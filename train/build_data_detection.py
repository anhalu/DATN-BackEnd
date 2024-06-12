import glob
import json
import os.path
import shutil

import cv2

from utils import get_image_from_pdf2


def get_image():
    output_dir = 'data/image/pdf_image'
    for file in glob.glob('/Volumes/thien/data_tc/data_raw/SongMai_TPBG/cotrongvilis_SongMai/*.pdf'):
        filename = os.path.basename(file)
        file_id, ext = os.path.splitext(filename)

        if len(glob.glob(f'data/image/pdf_image/{file_id}_*')) > 0:
            continue

        print(f"Process file: {file}")
        for i, image in enumerate(get_image_from_pdf2(file)):
            cv2.imwrite(f'{output_dir}/{file_id}_{i}.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])


def create_yolo_annotation(folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    else:
        shutil.rmtree(output_folder)
    for d in ['test', 'train', 'valid']:
        os.makedirs(os.path.join(output_folder, d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, d, 'labels'), exist_ok=True)

    num_sample = len(glob.glob(f'{folder}/*.json'))

    test_size = int(0.1 * num_sample)
    valid_size = int(0.1 * num_sample)
    train_size = num_sample - test_size - valid_size

    class_map = {}

    for i, label in enumerate(glob.glob(f'{folder}/*.json')):
        if i < train_size:
            output_dir = os.path.join(output_folder, 'train')
        elif i < train_size + valid_size:
            output_dir = os.path.join(output_folder, 'valid')
        else:
            output_dir = os.path.join(output_folder, 'test')

        label_data = json.load(open(label))
        width, height = label_data['imageWidth'], label_data['imageHeight']
        img_path = os.path.join(folder, label_data['imagePath'])
        shutil.copy(img_path, os.path.join(output_dir, "images", label_data['imagePath']))

        file_id, ext = os.path.splitext(label_data['imagePath'])
        label_file = os.path.join(output_dir, 'labels', f'{file_id}.txt')
        with open(label_file, 'w') as f:
            shapes = label_data['shapes']
            for shape in shapes:
                l = shape['label']
                if l not in class_map:
                    class_map[l] = len(class_map)
                points = shape['points']
                shape_type = shape['shape_type']
                if shape_type == 'rectangle':
                    x1, y1 = points[0]
                    x2, y2 = points[1]

                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2

                    ow, oh = x2 - x1, y2 - y1
                    f.write(f'{class_map[l]} {xc / width} {yc / height} {ow / width} {oh / height}\n')

    with open(os.path.join(output_folder, 'data.yaml'), 'w') as f:
        f.write("path: ../datasets/page_layout/\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")

        f.write(f"nc: {len(class_map)}\n")
        f.write(f"names:\n")
        for c, ci in class_map.items():
            f.write(f"  {ci}: {c}\n")



""" script training
export COMET_API_KEY=qomkokiNd4xV5cwtAD70edeN4
yolo detect train data=./datasets/page_layout/data.yaml model=yolov8n.pt epochs=10 lr0=0.01 imgsz=640 resume=True cache=True project=page_layout

"""
if __name__ == '__main__':
    create_yolo_annotation('/Volumes/thien/data_tc/training/pdf_image', '/Volumes/thien/data_tc/training/page_layout')
