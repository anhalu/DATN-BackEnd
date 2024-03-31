import os
import os.path
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.cuda
from loguru import logger
from paddleocr.paddleocr import parse_args
from paddleocr.tools.infer.predict_rec import TextRecognizer
from ultralytics import YOLO  # word detection

from layout_parser import DocumentBuilder
from layout_parser.elements import Document
from reader.base_reader import BaseReader

sys.path.append(os.getcwd())


class CusYOLOPaddleReader(BaseReader):
    def __init__(self):
        super().__init__()
        self.preserve_aspect_ratio = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.det_model = YOLO(f'{dir_path}/models/yolov8n_word_det.pt')

        # self.rec_model = PaddleOCR(lang='vi', rec_model_dir='models/paddle_param', rec_char_dict_path = 'models/word_dict.txt', wramup=True,gpu_mem=3000 ,rec_batch_num=100, recovery = True)

        params = parse_args(mMain=False)
        # params.rec_model_dir = './models/paddle_param'
        # params.rec_char_dict_path = f'./models/word_dict.txt'

        params.rec_model_dir = f'{dir_path}/../models/paddle_param'
        params.rec_char_dict_path = f'{dir_path}/../models/word_dict.txt'
        params.rec_batch_num = 400

        params.lang = 'vi'
        # params.rec_image_shape = "3,32,320"

        self.rec_model = TextRecognizer(params)
        self.layout_model = YOLO(f'{dir_path}/../models/layout_predict/predict_layout.pt')

        self.document_builder = DocumentBuilder(paragraph_break=0.025)
        self.score_threshold = 0.2

    def read_batches(self, images: List[np.ndarray], debug=False) -> Tuple[List[str], List[float]]:
        results, exe_time = self.rec_model(images)
        texts = []
        probs = []
        for result in results:
            texts.append(result[0])
            probs.append(result[1])

        if debug:
            logger.debug(f'Rec result: \n {texts}')
        return texts, probs

    def read_images(self, images: List[np.ndarray], return_merge=True) -> Document:
        data = {
            "boxes": [],
            "text_preds": [],
            "page_shapes": [],
            "orientations": None,
            "languages": None
        }
        det_res = self.det_model(images, imgsz=800, max_det=3000, iou=0.3, conf=0.3)
        img_boxes = []
        word_boxes = []
        split_page_img_boxes = []
        list_h_w = []
        for img_idx, (image, preds) in enumerate(zip(images, det_res)):
            for x1, y1, x2, y2 in preds.boxes.xyxy.to(torch.int32).tolist():
                word_boxes.append([x1, y1, x2, y2])
                img_boxes.append(image[y1: y2, x1:x2])
                split_page_img_boxes.append(img_idx)
                h, w = image.shape[:2]
                list_h_w.append((h, w))
        start_time_rec = time.time()
        texts, probs = self.read_batches(images=img_boxes)
        print(f"Time rec {len(img_boxes)} images: {time.time() - start_time_rec}")
        filtered_boxes, filtered_img_boxes, filtered_texts, filtered_probs = [], [], [], []
        cnt_img = 0
        for box_idx, (box, img_box, text, prob) in enumerate(zip(word_boxes, img_boxes, texts, probs)):
            if split_page_img_boxes[box_idx] == cnt_img:
                if prob >= self.score_threshold:
                    filtered_boxes.append(box)
                    filtered_img_boxes.append(img_box)
                    filtered_texts.append(text)
                    filtered_probs.append(prob)
            if split_page_img_boxes[box_idx] > cnt_img or box_idx == len(img_boxes) - 1:
                cnt_img += 1
                filtered_boxes = np.array(filtered_boxes)
                data['boxes'].append(filtered_boxes)
                data['text_preds'].append(list(zip(filtered_texts, filtered_probs)))
                data['page_shapes'].append(list_h_w[box_idx - 1])
                filtered_boxes, filtered_img_boxes, filtered_texts, filtered_probs = [], [], [], []
                if prob >= self.score_threshold:
                    filtered_boxes.append(box)
                    filtered_img_boxes.append(img_box)
                    filtered_texts.append(text)
                    filtered_probs.append(prob)
        # print(data)
        document = self.document_builder(**data)
        return document

    def predict_layout(self, images: List[np.ndarray]):
        response = {}
        results = self.layout_model(images)
        for img_idx, image, result in zip(range(len(images)), images, results):
            response[img_idx] = {"status": False, "title": "", "coordinates_title": [], "number": None,
                                 "coordinates_number": []}
            boxes = result.boxes.xyxy.to(torch.int32).tolist()
            classes = result.boxes.cls.to(torch.int32).tolist()
            probs = result.boxes.conf.cpu().tolist()
            max_prob_number = 0
            box_number = []
            max_prob_title = 0
            box_title = []

            for cls, prob, box in zip(classes, probs, boxes):
                if cls == 2:
                    if prob > max_prob_number:
                        max_prob_number = prob
                        box_number = box

                if cls == 4:
                    if prob > max_prob_title:
                        max_prob_title = prob
                        box_title = box
            if max_prob_number > 0.3:
                crop_img = image[box_number[1]: box_number[3], box_number[0]: box_number[2]]
                results, _ = self.rec_model([crop_img])
                response[img_idx]['number'] = results[0][0]
                response[img_idx]['coordinates_number'] = box_number
            if max_prob_title > 0.8:
                response[img_idx]['status'] = True
                response[img_idx]['title'] = ""
                response[img_idx]['coordinates_title'] = box_title
        return response


if __name__ == '__main__':
    list_img = []
    import glob

    for file in glob.glob(
            '/home/anhalu/anhalu-data/github/ocr_general_core/data/image/requests/3f49d2d9-9538-48b2-bf13-a46a5efd715e_*.jpg'):
        img = cv2.imread(file)
        list_img.append(img)
    model = CusYOLOPaddleReader()
    # res = model.read_images(list_img)
    # cnt_word = 0
    # import cv2

    # for page in res.pages:
    #     cnt = 0
    #     for block in page.blocks:
    #         for line in block.lines:
    #             for word in line.words:
    #                 cnt += 1
    #                 bb = word.bbox
    #                 cv2.rectangle(list_img[int(page.page_idx)], (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]),
    #                               color=(255, 0, 0), thickness=1)
    #                 cnt_word += 1

    #     print('page : ', page.page_idx, cnt)

    # print("WORD : ", cnt_word)
    print(model.predict_layout(list_img))

    # for i, img in enumerate(list_img):
    #     cv2.imwrite(f'{i}.jpg', img)
