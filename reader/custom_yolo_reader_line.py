import os
import os.path
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt

from layout_parser.elements import Word, Document

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


class CusYOLOReaderLine(BaseReader):
    def __init__(self):
        super().__init__()
        self.preserve_aspect_ratio = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.det_model = YOLO(f'{dir_path}/models/yolov8n_word_det.pt')

        reader_algo = config("READER_ALGO", default="vgg_seq2seq")
        rec_config = Cfg.load_config_from_name(reader_algo)
        rec_config['device'] = device
        rec_config['cnn']['pretrained'] = True
        rec_config['predictor']['beamsearch'] = False

        # rec_config['weights'] = f'{dir_path}/models/vgg_seq2seq.pth'

        self.rec_model = Predictor(rec_config)

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
        det_res = self.det_model(images, imgsz=800, max_det=3000, iou=0.3, conf=0.3)
        for img, preds in zip(images, det_res):
            word_boxes = preds.boxes.xyxy.to(torch.int32).tolist()
            img_boxes = [img[y1: y2 + 3, x1:x2 + 3] for x1, y1, x2, y2 in word_boxes]

            # texts, probs = self.read_batches(images=img_boxes)
            texts, probs = [None] * len(img_boxes), [0.7] * len(img_boxes)
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

        map_words = {}
        img_boxes = []
        idx = 0
        for page, img in zip(document.pages, images):
            for block in page.blocks:
                for line in block.lines:
                    new_word = Word(value="", bbox=line.bbox, confidence=0)
                    map_words[idx] = new_word
                    (x1, y1), (x2, y2) = line.bbox
                    img_boxes.append(img[y1:y2, x1:x2])
                    # plt.imshow(img_boxes[-1])
                    # plt.show()
                    line.words.clear()
                    line.words.append(new_word)
                    idx += 1

        texts, probs = self.read_batches(images=img_boxes)
        idx = 0
        for text, prob in zip(texts, probs):
            new_word = map_words.get(idx)
            new_word.value = text
            new_word.confidence = prob
            idx += 1

        return document


if __name__ == '__main__':
    img_path = '/Users/tienthien/workspace/tc_group/orc_general/ocr_core/raw_process/data/Bắc Sông Cấm TP HP 01.12.2023/2/20231201093905249_crop.jpg'
    image = cv2.imread(img_path)

    model = CusYOLOReader()
    res = model.read_images([image])
    # resu = res.export()
    # resu['content'] = res.render() 
    cnt_word = 0
    import cv2

    texts = []
    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(word.value)
                    texts.append(word.value)
                    bb = word.bbox
                    cv2.rectangle(image, (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]), color=(0, 255, 0), thickness=1)
                    # cv2.putText(image, word.value, (bb[0][0], bb[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=1)
                    cnt_word += 1
    with open("test1.txt", 'w', encoding="utf8") as f:
        f.write("\n".join(texts))

    # image = cv2.resize(image, (480, 640))     
    print("WORD : ", cnt_word)
    plt.imshow(image)
    plt.show()
    # cv2.imshow('test', image)
    # cv2.imwrite('anh1680.jpg', image)
    # cv2.waitKey(0)
