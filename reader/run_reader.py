from layout_parser import DocumentBuilder
from reader.table_recognizer import *
from reader.text_detector import *
from reader.text_direction import TextDirectionReader
from reader.text_recognizer import *


def cal_iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    # determine the coordinates of the intersection rectangle
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)
    # compute the area of intersection rectangle
    inter_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box1_area + box2_area - inter_area)
    # return the intersection over union value
    return iou


class Reader(BaseReader):
    def __init__(self, line_rec=True,
                 use_onnx=False,
                 table_rec=False):
        super().__init__()
        self.use_onnx = use_onnx
        self.line_rec = line_rec
        self.table_rec = table_rec

        self.det_model = YoloDet(model_name='yolov8m_word_det', version=3, score_threshold=0.5, iou=0.2,
                                 max_det=2000, img_size=1024)
        # self.line_det_model = PaddleDet(version=1)
        if self.line_rec:
            self.line_det_model = YoloDet(model_name='yolov8_line_det', version=1, img_size=1024, iou=0.3,
                                          score_threshold=0.5)
        else:
            self.line_det_model = None
        # self.hand_rec_model = PaddleRec(model_name="paddleocr_v4_hand", version=1)
        self.direction_model = TextDirectionReader(model_name='vgg19_bn', version=1, score_threshold=0.5)

        # old version -> 3 | only word
        if self.line_rec:
            self.rec_model = CustomPaddleRec(model_name='paddleocr_v4_en', version=1, batch_size=20)
        else:
            self.rec_model = PaddleRec(model_name='paddleocr_v4_en', version=1, batch_size=20)

        if self.table_rec:
            self.table_rec_model = TableRec(model_table_name='ppstructure_table', version_table=1,
                                            det_model=self.line_det_model, rec_model=self.rec_model)
        else:
            self.table_rec_model = None
        self.layout_model = LayoutYoloDet(model_name='layout_ocr_yolov8n', version=5, rec_model=self.rec_model,
                                          direction_model=self.direction_model, table_model=self.table_rec_model,
                                          img_size=640)

        self.document_builder = DocumentBuilder(paragraph_break=0.025)

        logger.info(f"Initialized Reader with line_rec={line_rec}; use_onnx={use_onnx}")

    def _detect_text(self, images: List[np.ndarray], file_id=None, *args, **kwargs) -> Document:
        det_res = self.det_model(images)
        data = {
            "boxes": [],
            "text_preds": [],
            "page_shapes": [],
            "orientations": None,
            "languages": None,
            "file_id": file_id
        }
        for page, image in zip(det_res, images):
            h, w = image.shape[:2]
            boxes = []
            for cls, cls_name, prob, box, crop_img in page:
                boxes.append(box)
            data['boxes'].append(np.array(boxes))
            data['page_shapes'].append((h, w))
            data['text_preds'].append([(None, 0)] * len(boxes))
        document = self.document_builder(**data)
        logger.debug(document.info)
        return document

    def _recognite_text(self, images: List[np.ndarray], *args, **kwargs):
        return self.rec_model(images)

    def read_images(self, images: List[np.ndarray], return_merge=True, file_id=None, *args, **kwargs) -> Document:
        document = self._detect_text(images, file_id=file_id)
        line_images = []
        word_images = []
        for page, image in zip(document.pages, images):
            for block in page.blocks:
                for line in block.lines:
                    (x1, y1), (x2, y2) = line.bbox
                    line_images.append(image[y1:y2, x1:x2])
                    for word in line.words:
                        (x1, y1), (x2, y2) = word.bbox
                        word_img = image[y1: y2, x1: x2]
                        word_images.append(word_img)

        coord_splits_all = None
        if self.line_rec:
            texts, probs, coord_splits_all = self._recognite_text(line_images)
        else:
            texts, probs, _ = self._recognite_text(word_images) 

        post_word_images = []
        post_words = []

        i = 0
        for page, image in zip(document.pages, images):
            for block in page.blocks:
                for line in block.lines:
                    if self.line_rec:
                        text = texts[i]
                        prob = probs[i]
                        coord_split_line = coord_splits_all[i]
                        split_text = text.split()
                        (x1, y1), (x2, y2) = line.bbox
                        last_start_word_x = x1
                        if len(split_text) == len(line.words):
                            for word_text, coord_split, word in zip(split_text, coord_split_line[1:], line.words):
                                x1_w, y1_w = last_start_word_x, y1
                                x2_w, y2_w = x1 + int(coord_split), y2
                                last_start_word_x = x2_w

                                word_box = word.bbox[0] + word.bbox[1]
                                if cal_iou((x1_w, y1_w, x2_w, y2_w), word_box) > 0.5:
                                    word.value = word_text
                                    word.confidence = prob
                                    # word.bbox = [[x1_w, y1_w], [x2_w, y2_w]]
                                else:
                                    # insert into post predict on word
                                    (x1, y1), (x2, y2) = word.bbox
                                    word_img = image[y1: y2, x1: x2]
                                    post_word_images.append(word_img)
                                    post_words.append(word)

                        else:
                            # insert into post predict on word
                            for word in line.words:
                                (x1, y1), (x2, y2) = word.bbox
                                word_img = image[y1: y2, x1: x2]
                                post_word_images.append(word_img)
                                post_words.append(word)

                        i += 1
                    else:
                        # processing on words result
                        for word in line.words:
                            if probs[i] >= self.rec_model.score_threshold:
                                word.value = texts[i]
                                word.confidence = probs[i]
                            i += 1
        # post predict for word not detected line
        if post_word_images:
            word_texts, word_probs, _ = self._recognite_text(post_word_images)
            for word, word_text, word_prob in zip(post_words, word_texts, word_probs):
                if word_prob >= self.rec_model.score_threshold:
                    word.value = word_text
                    word.confidence = word_prob

        # Clear empty line, box
        for page, image in zip(document.pages, images):
            for block in page.blocks:
                for line in block.lines:
                    line.words = list(filter(lambda x: x.value, line.words))
                block.lines = list(filter(lambda x: x.words, block.lines))
            page.blocks = list(filter(lambda x: x.lines, page.blocks))

        return document

    def predict_layout(self, images, pages) -> Dict:
        return self.layout_model(images, pages=pages)


def test_reader():
    # det_m = YoloDet(version=1, score_threshold=0.3, iou=0.3)
    # rec_m = PaddleRec(version=1)
    # hand_rec = PaddleRec(model_name="paddleocr_v3_hand", version=1)
    # r_m = Reader(det_m, rec_m, hand_rec, line_rec=False)
    r_m = Reader(line_rec=True)
    r_m.init_model()
    list_imgs = [cv2.imread(
        '/home/anhalu/anhalu-data/ocr_general_core/data/image/requests/bb3362fe-89f5-40a3-a5ca-0ceec1d96382.jpg')]
    doc = r_m.read_images(list_imgs)
    for page in doc.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    print(word.value)

    doc.show(list_imgs)
    exit(0)

    a = r_m.predict_layout(list_imgs)
    print(a)
    data = doc.export()
    data['content'] = doc.render()
    print(data['content'])

    print("num_ page break: ", str(data['content']).count('\n\n\n\n'))


if __name__ == '__main__':
    test_reader()
