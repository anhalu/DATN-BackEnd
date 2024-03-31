import copy

import cv2

from reader.base_reader import *
from bs4 import BeautifulSoup

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def remove_empty_rows_columns(html):
    soup = BeautifulSoup(html, 'html.parser')

    rows = soup.find_all('tr')
    # cells = [[cell for cell in row.find_all(['td', 'th'], recursive=False)] for row in rows]
    #
    # # Remove empty rows
    # for i, row in enumerate(rows):
    #     for j, cell in enumerate(cells[i]):
    #         if cell.text.strip() == "":
    #             cells[i][j].extract()
    for i, row in enumerate(rows):
        if row.td is not None:
            flag_rm = True
            for cell in row.find_all(['td', 'th']):
                if cell.text.strip() != '':
                    flag_rm = False
            if flag_rm:
                row.td.extract()
        if row.text is None or row.text.strip() == '':
            row.extract()
    return str(soup)


class TableRec(BaseTable):
    def __init__(self, model_table_name="ppstructure_table", version_table=1, det_model=None, rec_model=None):
        from paddleocr.paddleocr import parse_args
        from paddleocr.ppstructure.table.predict_structure import TableStructurer
        from paddleocr.ppstructure.table.matcher import TableMatch
        from paddleocr.ppstructure.table.table_master_match import TableMasterMatcher

        model_table_path = f'models/{model_table_name}_v{version_table}'
        super().__init__(model_table_path=model_table_path)

        # args
        args = parse_args(mMain=False)
        args.use_pdserving = False
        args.use_gpu = True
        args.lang = 'en'
        args.table_model_dir = self.model_table_path
        args.table_char_dict_path = os.path.join(self.dir_path, f'models/table_structure_dict.txt')
        args.show_log = False,
        args.layout = False
        args.merge_no_span_structure = True
        args.warmup = True

        self.table_model = TableStructurer(args)
        if args.table_algorithm in ['TableMaster']:
            self.match = TableMasterMatcher()
        else:
            self.match = TableMatch(filter_ocr_result=True)

        self.det_model = det_model
        self.rec_model = rec_model

    def _structure(self, image):
        structure, elapse = self.table_model(copy.deepcopy(image))
        return structure, elapse

    def predict(self, images: List[np.ndarray]):
        expand = 0
        det_results = self.det_model(copy.deepcopy(images), name='Line in Table')
        html_predictions = []
        for det_result, image in zip(det_results, images):
            dt_boxes = []
            for cls, cls_name, prob, box, crop_img in det_result:
                x_min, y_min, x_max, y_max = box
                dt_boxes.append([[x_min, y_min], [x_max, y_max]])
            dt_boxes = np.array(dt_boxes)
            dt_boxes_n = sorted_boxes(dt_boxes)
            dt_boxes = []
            line_images = []
            h, w = image.shape[:2]
            for box in dt_boxes_n:
                (x_min, y_min), (x_max, y_max) = box
                x_min = max(0, x_min - expand)
                y_min = max(0, y_min - expand)
                x_max = min(w, x_max + expand)
                y_max = min(h, y_max + expand)
                dt_boxes.append([x_min, y_min, x_max, y_max])
                line_images.append(image[y_min:y_max, x_min:x_max])
            texts, probs, _ = self.rec_model(line_images)
            rec_res = [(t, p) for t, p in zip(texts, probs)]
            html_pred = self.predict_html(image, dt_boxes, rec_res)
            html_predictions.append(html_pred)
        return html_predictions

    def predict_html(self, image, dt_boxes, rec_res):
        structure, _ = self._structure(copy.deepcopy(image))
        dt_boxes = np.array(dt_boxes)
        rec_res = np.array(rec_res)
        pred_html = self.match(structure, dt_boxes, rec_res)
        return remove_empty_rows_columns(pred_html)

    def __call__(self, images, *args, **kwargs):
        return self.predict(images)


def test_table():
    from text_detector import PaddleDet, YoloDet
    from text_recognizer import PaddleRec
    img = cv2.imread(
        '/home/anhalu/anhalu-data/word_det/checkbang.jpg')
    import time
    # det_model = PaddleDet(version=1)
    det_model = YoloDet(model_name='yolov8_line_det', version=1, img_size=640, iou=0.3,
                                      score_threshold=0.5)
    rec_model = PaddleRec(model_name='paddleocr_v4_all', version=2, batch_size=20)
    m = TableRec(det_model=det_model, rec_model=rec_model)
    start = time.time()
    all_html = m([img])
    print("TIME REC : ", time.time() - start)
    print(all_html)


if __name__ == '__main__':
    test_table()
