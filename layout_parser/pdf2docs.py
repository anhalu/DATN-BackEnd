import math

import cv2
import numpy as np

try:
    from .document_builder import DocumentBuilder
except:
    from document_builder import DocumentBuilder
import os
import json


class Pdf2Docs(DocumentBuilder):
    def __init__(self, request_id: str = None):
        super().__init__()
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if not request_id.endswith('.json'):
            request_id += '.json'
        if not request_id.startswith('/'):
            request_id = f'data/image/requests/{request_id}'
        self.request_id = request_id

    def sortboxes(self, boxes: np.ndarray):
        idxs, boxes = self._sort_boxes(boxes)
        return idxs, boxes

    def run(self):
        with open(self.request_id, 'r') as f:
            data = json.load(f)
        filename = os.path.basename(self.request_id)
        dirname = os.path.dirname(self.request_id)
        name = os.path.splitext(filename)[0]
        results_all_page = ""
        pages = data['pages']
        num_page = len(pages)
        images = [cv2.imread(os.path.join(dirname, name) + f'_{i}.jpg') for i in range(num_page)]
        min_x = -1
        for page, image in zip(pages, images):
            list_line = []
            list_value = []
            list_h_word = []
            for block in page['blocks']:
                for line in block['lines']:
                    (x1, y1), (x2, y2) = line['bbox']
                    list_line.append([x1, y1, x2, y2])
                    tmp = []
                    for word in line['words']:
                        (x1, y1), (x2, y2) = word['bbox']
                        if min_x == -1:
                            min_x = x1
                        else:
                            min_x = min(min_x, x1)
                        tmp.append(word['value'])
                        list_h_word.append(abs(y2 - y1))
                    list_value.append(' '.join(tmp))
            if len(list_line) == 0:
                continue
            list_line = np.array(list_line)
            list_h_word = np.array(list_h_word)
            list_value = np.array(list_value)
            space_distance = np.median(list_h_word) / 2

            y_center_line = -1
            previous_x = -1
            previous_y = -1
            for idx_line, value in zip(range(len(list_line)), list_value):
                if y_center_line < 0:
                    # cnt += 1
                    y_center_line = (list_line[idx_line][1] + list_line[idx_line][3]) / 2
                    # dict_line[cnt].append(list_line[idx_line])
                    num_space = int((list_line[idx_line][0] - min_x) / space_distance)
                    add = num_space * ' ' + value + ' '
                    results_all_page += add
                    previous_x = list_line[idx_line][2]
                    previous_y = list_line[idx_line][3]
                else:
                    y_current_line = (list_line[idx_line][1] + list_line[idx_line][3]) / 2
                    if list_line[idx_line][0] < previous_x or (
                            abs(y_current_line - y_center_line) > 2 and y_current_line > (
                            previous_y + y_center_line) / 2):
                        num_line = int((y_current_line - y_center_line) // (space_distance * 2.5))
                        # print(num_line)
                        if num_line < 1:
                            num_line = 1
                        results_all_page += num_line * '\n'
                        previous_x = list_line[idx_line][2]
                        previous_y = list_line[idx_line][3]
                        num_space = int((list_line[idx_line][0] - min_x) / space_distance)
                        add = num_space * ' ' + value + ' '
                        results_all_page += add
                        y_center_line = y_current_line
                    else:
                        num_space = int(abs((list_line[idx_line][0] - previous_x)) / space_distance)
                        previous_x = list_line[idx_line][0]
                        add = num_space * ' ' + value + ' '
                        results_all_page += add
                # print(num_space)
            results_all_page += '\n\n\n\n\n\n\n\n\n\n\n'
        return results_all_page


if __name__ == '__main__':
    t = Pdf2Docs(
        '/home/anhalu/anhalu-data/github/ocr_general_core/data/image/requests/c06c926b-50eb-48a9-bac0-e08b7d8cdb9d.json')
    res = t.run()
    with open('save_test.txt', 'w', encoding='utf-8') as file:
        file.write(res)
