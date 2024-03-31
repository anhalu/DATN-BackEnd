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
from loguru import logger
from mmocr.apis import MMOCRInferencer
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import utils
from layout_parser import DocumentBuilder
from reader.base_reader import BaseReader


class CusMMocrReader(BaseReader):
    def __init__(self):
        super().__init__()
        self.preserve_aspect_ratio = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.det_model = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015')
        '''
        dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015
        '''

        # self.det_model.forward_kwargs.add("pred_score_thr")

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
        det_res = self.det_model(images, return_vis=False)

        for img, det in zip(images, det_res['predictions']):
            word_boxes = np.array(det['det_polygons'])
            word_conf = np.array(det['det_scores'])

            word_boxes = word_boxes.astype(np.int32)

            xmin = np.min(word_boxes[:, [0, 2, 4, 6]], axis=1)
            xmax = np.max(word_boxes[:, [0, 2, 4, 6]], axis=1)

            ymin = np.min(word_boxes[:, [1, 3, 5, 7]], axis=1)
            ymax = np.max(word_boxes[:, [1, 3, 5, 7]], axis=1)

            word_boxes_n = []
            img_boxes = []
            font = cv2.FONT_HERSHEY_SIMPLEX
            for idx in range(len(xmin)):
                if word_conf[idx] < 0.6:
                    continue
                word_boxes_n.append([xmin[idx], ymin[idx], xmax[idx], ymax[idx]])
                img_boxes.append(img[ymin[idx]:ymax[idx], xmin[idx]: xmax[idx]])
                print(xmin[idx])
                cv2.rectangle(img, (xmin[idx], ymin[idx]), (xmax[idx], ymax[idx]), color=(0, 255, 0))
                cv2.putText(img=img, text=f"{word_conf[idx]:.2f}", org=(int(xmin[idx]), int(ymin[idx])), fontFace=font,
                            fontScale=0.3, color=(255, 0, 0), thickness=1)
            cv2.imshow('test', img)
            cv2.waitKey(0)

            texts, probs = self.read_batches(images=img_boxes)

            filtered_boxes, filtered_img_boxes, filtered_texts, filtered_probs = [], [], [], []

            for box, img_box, text, prob in zip(word_boxes_n, img_boxes, texts, probs):
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
    img_path = '/home/anhalu/anhalu-data/github/ocr_general_core/data/image/requests/82384163-850d-4df5-9024-f8e4bf2247bc_5.jpg'
    # test(img_path)
    # exit()
    image = cv2.imread(img_path)
    # image = cv2.resize(image, (640, 640))
    model = CusMMocrReader()
    res = model.read_images([image])
    # resu = res.export()
    # resu['content'] = res.render() 
    # import cv2 
    # for page in res.pages: 
    #     for block in page.blocks: 
    #         for line in block.lines : 
    #             for word in line.words :   
    #                 print(word) 
    #                 bb = word.bbox 
    #                 cv2.rectangle(image, (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]), color=(255, 0,0), thickness = 1) 

    # cv2.imshow('test', image) 
    # cv2.waitKey(0)
