import os
import os.path
import sys
from typing import List, Tuple

from layout_parser.elements import Document

sys.path.append(os.getcwd())

import numpy as np
import torch.cuda
from PIL import Image
import cv2
from decouple import config

from ultralytics import YOLO
from loguru import logger
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import utils
from layout_parser import DocumentBuilder
from reader.base_reader import BaseReader


class CusYOLOReader(BaseReader):
    def __init__(self):
        super().__init__()
        self.preserve_aspect_ratio = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.det_model = YOLO(f'{dir_path}/models/yolov8n_word_det.pt')

        reader_algo = config("READER_ALGO", default="vgg_seq2seq")
        rec_config = Cfg.load_config_from_name(reader_algo)
        rec_config['device'] = 'cuda:0'
        rec_config['cnn']['pretrained'] = False
        rec_config['predictor']['beamsearch'] = True

        rec_config['weights'] = f'{dir_path}/models/vgg_seq2seq.pth'

        self.rec_model = Predictor(rec_config)

        self.layout_model = YOLO('models/layout_predict/predict_layout.pt')

        self.document_builder = DocumentBuilder()
        self.score_threshold = 0.2

    def read_batches(self, images: List[np.ndarray], debug=False) -> Tuple[List[str], List[float]]:
        imgs = [Image.fromarray(i) for i in images]
        res = self.rec_model.predict_batch(imgs, return_prob=True)
        if debug:
            logger.debug(f'Rec result: \n {res}')
        return res

    def read_batches_box(self, image: np.ndarray, boxes: List, debug=False) -> Tuple[
        List, Tuple[List[str], List[float]]]:
        box_imgs = []
        for l in boxes:
            img_line, _ = utils.crop_4_points(image, l)
            box_imgs.append(img_line)
        return box_imgs, self.read_batches(images=box_imgs, debug=debug)

    def read_images(self, images: List[np.ndarray], return_merge=True) -> Document:
        data = {
            "boxes": [],
            "text_preds": [],
            "page_shapes": [],
            "orientations": None,
            "languages": None
        }
        det_res = self.det_model(images, imgsz=640, max_det=3000, iou=0.3, conf=0.3)
        for img, preds in zip(images, det_res):
            word_boxes = preds.boxes.xyxy.to(torch.int32).tolist()
            img_boxes = [img[y1: y2 + 1, x1 + 1:x2 + 2] for x1, y1, x2, y2 in word_boxes]

            texts, probs = self.read_batches(images=img_boxes)
            filtered_boxes, filtered_img_boxes, filtered_texts, filtered_probs = [], [], [], []

            for box, img_box, text, prob in zip(word_boxes, img_boxes, texts, probs):
                if prob >= self.score_threshold:
                    filtered_boxes.append(box)
                    filtered_img_boxes.append(img_box)
                    filtered_texts.append(text)
                    filtered_probs.append(prob)
            h, w = img.shape[:2]
            filtered_boxes = np.array(filtered_boxes)
            data['boxes'].append(filtered_boxes)
            data['text_preds'].append(list(zip(filtered_texts, filtered_probs)))
            data['page_shapes'].append((h, w))
        document = self.document_builder(**data)
        return document

    def predict_layout(self, images: List[np.ndarray]):
        response = {}
        results = self.layout_model(images)
        for img_idx, image, result in zip(range(len(images)), images, results):
            response[img_idx] = {"status": False, "title": "", "coordinates_title": [], "number": None,
                                 "coordinates_number": []}
            boxes = result.boxes.xyxy.to(torch.int32).tolist()
            clses = result.boxes.cls.to(torch.int32).tolist()
            probs = result.boxes.conf.cpu().tolist()
            for cls, prob, box in zip(clses, probs, boxes):
                if cls == 2:
                    if prob < 0.3:
                        continue
                    crop = Image.fromarray(image[box[1]: box[3], box[0]: box[2]])
                    response[img_idx]['number'] = self.rec_model.predict(crop)
                    response[img_idx]['coordinates_number'] = box

                if cls == 4:
                    if prob < 0.5:
                        continue
                    response[img_idx]['status'] = True
                    response[img_idx]['title'] = ""
                    response[img_idx]['coordinates_title'] = box
        return response


if __name__ == '__main__':

    list_img = []

    import glob

    for file in glob.glob(
            '/home/anhalu/anhalu-data/github/ocr_general_core/data/image/requests/efb7e004-b79e-4887-9da0-bc4b8c25e2d9*.jpg'):
        img = cv2.imread(file)
        list_img.append(img)
    model = CusYOLOReader()
    res = model.read_images(list_img)

    cnt_word = 0
    import cv2

    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    bb = word.bbox
                    cv2.rectangle(list_img[int(page.page_idx)], (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]),
                                  color=(255, 0, 0), thickness=1)
                    cnt_word += 1

    print("WORD : ", cnt_word)

    for i, img in enumerate(list_img):
        cv2.imwrite(f'{i}.jpg', img)
