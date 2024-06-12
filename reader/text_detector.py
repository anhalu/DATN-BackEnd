import time
from typing import Dict

import cv2
import torch
from PIL import Image

from reader.base_reader import *


class YoloDet(BaseDet):
    def __init__(self, model_name='yolov8n_word_det', version=1, score_threshold=0.01, img_size=640, iou=0.5,
                 max_det=None):
        from ultralytics import YOLO
        model_path = f'models/{model_name}_v{version}.pt'
        if max_det is None:
            max_det = 500
        super().__init__(model_path, score_threshold, img_size, iou, max_det=max_det)
        self.model = YOLO(self.model_path)

    def __call__(self, images: List[np.ndarray], return_image=False, name='Word Detection', *args, **kwargs):
        start_time = time.time()
        conf = 0.01
        if isinstance(self.score_threshold, float):
            conf = self.score_threshold
        det_res = self.model(images, imgsz=self.img_size, max_det=self.max_det, iou=self.iou, conf=conf)
        res = []
        count_obj = 0
        for img_idx, (image, result) in enumerate(zip(images, det_res)):
            boxes = result.boxes.xyxy.to(torch.int32).tolist()
            classes = result.boxes.cls.to(torch.int32).tolist()
            probs = result.boxes.conf.cpu().tolist()
            name_dict = result.names
            # print(name_dict)
            img_res = []
            for cls, prob, box in zip(classes, probs, boxes):
                class_name = name_dict[cls]
                flag_valid = True
                if isinstance(self.score_threshold, Dict):
                    score_thresh = self.score_threshold.get(class_name, 0)
                    if prob < score_thresh:
                        flag_valid = False
                else:
                    if prob < self.score_threshold:
                        flag_valid = False

                if flag_valid:
                    crop_img = None
                    if return_image:
                        crop_img = image[box[1]: box[3], box[0]: box[2]]
                    img_res.append((cls, class_name, prob, box, crop_img))
                    count_obj += 1

            res.append(img_res)
        end_time = time.time()
        logger.info(f"{name} Detect {len(images)} images: return {count_obj} objects in {end_time - start_time:.5}s")
        return res


class LayoutYoloDet(YoloDet):
    def __init__(self, model_name, version, img_size=640, iou=0.5, rec_model=None, direction_model=None,
                 table_model=None):
        score_threshold = {
            'title': 1.0,
            'so_to': 1.0,
            'table': 1.0,
            'figure': 1.0,
            'signature': 1.0,
        }
        super().__init__(model_name, version, score_threshold, img_size, iou)
        self.rec_model = rec_model
        self.direction_model = direction_model
        self.table_model = table_model

    def __call__(self, images: List[np.ndarray], return_image=True, pages=None, *args, **kwargs):
        det_res = super().__call__(images, return_image, name="Layout Detection")

        response = {}
        soto_images = []
        list_idx_soto = []
        table_images = []
        list_idx_table = []
        list_box_table = []
        for img_idx, (image, det) in enumerate(zip(images, det_res)):
            h, w = image.shape[:2]
            response[img_idx] = {
                "status": False,
                "title": "",
                "title_prob": 0,
                "coordinates_title": [],
                "number": None,
                "number_prob": 0,
                "coordinates_number": [],
                "table": [],
                "coordinates_table": [],
                'signatures': [],
                'coordinates_signature': [],
                'signature_prob': [],

                "dimensions": [w, h],
                "figures": [],
                "tables": [],
                "titles": None,
                "page_numbers": None,
                "signature_boxes": [],

            }
            soto_image = None
            for cls, cls_name, prob, box, crop_img in det:
                x1, y1, x2, y2 = box
                if cls_name == 'signature':
                    response[img_idx]['coordinates_signature'].append([x1, y1, x2, y2])
                    response[img_idx]['signature_prob'].append(prob)
                if cls_name == 'title' and prob >= response[img_idx]['title_prob']:
                    response[img_idx]['status'] = True
                    response[img_idx]['coordinates_title'] = box
                    response[img_idx]['title_prob'] = prob
                    response[img_idx]['titles'] = box
                if cls_name == 'so_to' and prob >= response[img_idx]['number_prob']:
                    soto_image = Image.fromarray(crop_img)
                    response[img_idx]['coordinates_number'] = box
                    response[img_idx]['number_prob'] = prob
                    response[img_idx]['page_numbers'] = box
                if cls_name == 'table':
                    x1 = max(x1 - 3, 0)
                    y1 = max(y1 - 3, 0)
                    x2 = min(x2 + 3, w)
                    y2 = min(y2 + 3, h)
                    response[img_idx]['tables'].append((x1, y1, x2, y2))

                    crop_img = image[y1: y2, x1: x2]
                    h_crop, w_crop = crop_img.shape[:2]
                    n_h = 640
                    n_w = (w_crop * 640) // h_crop
                    crop_img = cv2.resize(crop_img, (n_w, n_h), interpolation=cv2.INTER_LINEAR)
                    table_images.append(crop_img)
                    list_idx_table.append(img_idx)
                    list_box_table.append(box)
                if cls_name == 'figure':
                    response[img_idx]['figures'].append(box)
                if cls_name == 'signature':
                    response[img_idx]['signature_boxes'].append(box)

            if soto_image is not None:
                soto_images.append(soto_image)
                list_idx_soto.append(img_idx)

        if len(soto_images) > 0:
            batch_soto = len(soto_images)
            soto_images = [np.array(i) for i in self.direction_model.rotate(soto_images, batch_soto)]
            rec_texts, rec_probs, _ = self.rec_model(soto_images)
            for idx, text in zip(list_idx_soto, rec_texts):
                response[idx]['number'] = text

        if len(table_images) > 0:
            start = time.time()
            if self.table_model:
                table_htmls = self.table_model(table_images)
            else:
                table_htmls = [None] * len(table_images)
            for idx, table_html, box in zip(list_idx_table, table_htmls, list_box_table):
                response[idx]['table'].append(table_html)
                response[idx]['coordinates_table'].append(box)
            logger.info(f"Table {len(table_images)}: return {len(table_htmls)} in {time.time() - start}")
        return response


class PaddleDet(BaseDet):
    def __init__(self, model_name='paddleocr_v3_det', version=1, score_threshould=0.2, img_size=640):
        from paddleocr.paddleocr import parse_args
        from paddleocr.tools.infer.predict_det import TextDetector

        model_path = f'models/{model_name}_v{version}'
        super().__init__(model_path, score_threshold=score_threshould, img_size=img_size)

        params = parse_args(mMain=False)
        params.det_model_dir = self.model_path
        params.lang = 'vi'
        params.use_gpu = True
        params.det_algorithm = 'DB++'
        params.det_db_unclip_ratio = 1.0
        params.det_db_box_thresh = 0.6
        self.model = TextDetector(params)

    def __call__(self, images, return_image=False, *args, **kwargs):
        det_results = []
        for image in images:
            prediction, exec_time = self.model(image)
            # res = []
            res_img = []
            for box in prediction:
                box = box.astype('int32')
                (x_min, y_min), (x_max, y_max) = np.min(box, axis=0), np.max(box, axis=0)
                crop_img = None
                if return_image:
                    crop_img = image[box[1]: box[3], box[0]: box[2]]
                res_img.append((0, 'text', 0.0, [x_min, y_min, x_max, y_max], crop_img))
            det_results.append(res_img)
        return det_results


def test_padde_det():
    img = cv2.imread('/home/anhalu/anhalu-data/ocr_general_core/t1.jpg')
    m = PaddleDet()
    pred = m(img)
    print(pred)


def test_yolo_det():
    score_threshold = {
        'title': 0.7,
        'so_to': 0.2,
        'table': 0.2,
        'figure': 0.1,
        'signature': 0.1
    }
    m = LayoutYoloDet()


def test_yolo_det_word():
    import matplotlib.pyplot as plt
    from PIL import ImageDraw, Image
    det_model = YoloDet(version=2, score_threshold=0.5, iou=0.5, max_det=3000)
    img = cv2.imread('../data/image/requests/addac693-3c68-4a2e-9a83-b9cf9ab46b3e_0.jpg')
    pred = det_model([img])[0]

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for _, c, p, box, _ in pred:
        x1, y1, x2, y2 = box
        draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline=(0, 0, 255, 30))

    plt.imshow(img_pil)
    plt.show()


if __name__ == '__main__':
    test_yolo_det_word()
