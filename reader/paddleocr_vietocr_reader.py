import os
import sys

sys.path.append(os.getcwd())
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from decouple import config
from doctr.io import Document
from loguru import logger

try:
    from paddleocr import PaddleOCR
except Exception:
    pass
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import utils
from layout_parser import DocumentBuilder
from reader.base_reader import BaseReader


class PaddleVietReader(BaseReader):
    def __init__(self):
        super().__init__()
        lang = config("LANGUAGE", default="en")
        det_model_dir = config("DET_MODEL_DIR", default="./models/en_PP-OCRv3_det_infer")
        det_algorithm = config("DET_ALGORITHM", default="DB")
        det_db_thresh = config("DET_DB_THRESH", default=0.3, cast=float)
        det_db_box_thresh = config("DET_DB_BOX_THRESH", default=0.5, cast=float)
        det_db_unclip_ratio = config("DET_DB_UNCLIP_RATIO", default=2, cast=float)
        use_gpu = config("USE_GPU", default=False, cast=bool)

        reader_algo = config("READER_ALGO", default="vgg_seq2seq")
        self.det_model = PaddleOCR(use_angle_cls=True,
                                   lang=lang,
                                   det_algorithm=det_algorithm,
                                   det_model_dir=det_model_dir,
                                   det_db_thresh=det_db_thresh,
                                   det_db_box_thresh=det_db_box_thresh,
                                   det_db_unclip_ratio=det_db_unclip_ratio,
                                   use_gpu=use_gpu)

        rec_config = Cfg.load_config_from_name(reader_algo)
        rec_config['device'] = 'cuda:0' if use_gpu else 'cpu'
        rec_config['cnn']['pretrained'] = True
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # rec_config['weights'] = f'{dir_path}/models/vgg_seq2seq.pth'
        rec_config['predictor']['beamsearch'] = True
        self.rec_model = Predictor(rec_config)

        self.document_builder = DocumentBuilder()
        self.score_threshold = 0.3

    @utils.timeit
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
        for image in images:
            det_res = self.det_model.ocr(image, cls=True, det=True, rec=False)[0]
            img_boxes, rec_result = self.read_batches_box(image, boxes=det_res)
            texts, probs = rec_result

            filtered_boxes, filtered_img_boxes, filtered_texts, filtered_probs = [], [], [], []

            for box, img_box, text, prob in zip(det_res, img_boxes, texts, probs):
                if prob >= self.score_threshold:
                    filtered_boxes.append(box)
                    filtered_img_boxes.append(img_box)
                    filtered_texts.append(text)
                    filtered_probs.append(prob)

            h, w = image.shape[:2]
            boxes = np.array(filtered_boxes)
            data['boxes'].append(boxes)
            data['text_preds'].append(list(zip(filtered_texts, filtered_probs)))
            data['page_shapes'].append((h, w))

        document = self.document_builder(**data)
        return document


if __name__ == '__main__':
    model = PaddleVietReader()
    img_path = '/home/anhalu/anhalu-data/github/ocr_general_core/data/image/requests/f9c72cba-3ea1-49ab-9a26-15b2aeb61491_2.jpg'
    img = cv2.imread(img_path)
    res = model.read_images([img])
    print(res.show([img]))
