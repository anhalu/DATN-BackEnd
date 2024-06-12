import os
import os.path
import sys
from typing import List, Tuple

from layout_parser.elements import Document

sys.path.append(os.getcwd())

import numpy as np
import torch.cuda
from PIL import Image
from decouple import config
from doctr.io import read_img_as_numpy
from doctr.models import detection_predictor
from loguru import logger
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import utils
from layout_parser import DocumentBuilder
from reader.base_reader import BaseReader


class CusDocTrReader(BaseReader):
    def __init__(self):
        super().__init__()
        self.preserve_aspect_ratio = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.det_model = detection_predictor(arch='db_resnet50',
                                             pretrained=True,
                                             assume_straight_pages=True,
                                             preserve_aspect_ratio=self.preserve_aspect_ratio).to(device)
        use_gpu = config("USE_GPU", default=True, cast=bool)
        reader_algo = config("READER_ALGO", default="vgg_seq2seq")
        rec_config = Cfg.load_config_from_name(reader_algo)
        rec_config['device'] = 'cuda:0' if use_gpu else 'cpu'
        rec_config['cnn']['pretrained'] = False
        rec_config['predictor']['beamsearch'] = True

        dir_path = os.path.dirname(os.path.realpath(__file__))
        rec_config['weights'] = f'{dir_path}/models/vgg_seq2seq.pth'

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
        det_res = self.det_model(images)
        for img, page in zip(images, det_res):
            h, w = img.shape[:2]
            rw = max(h, w) if self.preserve_aspect_ratio else w
            rh = max(h, w) if self.preserve_aspect_ratio else h
            word_boxes = page['words']
            if word_boxes.shape[-1] == 2:
                word_boxes[:, :, 0] = word_boxes[:, :, 0] * rw
                word_boxes[:, :, 1] = word_boxes[:, :, 1] * rh
                word_boxes = word_boxes.astype(np.int32)
                img_boxes = [utils.crop_4_points(img, b)[0] for b in word_boxes]
            elif word_boxes.shape[-1] == 5:
                word_boxes[:, 0] *= rw
                word_boxes[:, 1] *= rh
                word_boxes[:, 2] *= rw
                word_boxes[:, 3] *= rh
                word_boxes = word_boxes[:, :4].astype(np.int32)
                img_boxes = [img[b[1]: b[3], b[0]: b[2]] for b in word_boxes]
            else:
                raise NotImplementedError()

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


if __name__ == '__main__':
    img_path = '/home/anhalu/anhalu-data/github/ocr_general_core/data/image/requests/0baa071d-e803-40cc-a13c-6e1d351d031d_0.jpg'
    image = read_img_as_numpy(img_path)
    model = CusDocTrReader()
    res = model.read_images([image])
    # resu = res.export()
    # resu['content'] = res.render() 
    import cv2

    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    bb = word.bbox
                    cv2.rectangle(image, (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]), color=(255, 0, 0), thickness=1)

    cv2.imshow('test', image)
    cv2.waitKey(0)
    # res.show([image])
