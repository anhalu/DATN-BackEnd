import copy
import json
import re
import os
from pathlib import Path
from typing import List, Optional

import cv2

from application.schemas import RuleNameEnum
from content_parser.basic_rule_doc import *
from content_parser import utils
from layout_parser.elements import Document

# from content_parser.rule_v7 import rule_mapping
rule_mapping = {
    RuleNameEnum.PREFIX: prefix_rule,
    RuleNameEnum.SUFFIX: suffix_rule,
    RuleNameEnum.REGEX: regex_rule,
    RuleNameEnum.CONTAINS: contain_rule,
    RuleNameEnum.EXACTLY: exactly_rule,
    RuleNameEnum.PAGE_IDX: page_idx_rule,
    RuleNameEnum.BOUNDING_BOX: bounding_box_rule
}
model = None
BASE_DIR = Path(__file__).resolve().parent.parent


def custom_rule_gcnqsdd(document: Document, content: str, start_idx, end_idx,
                        last_matches, field_mapping=None,
                        **kwargs) -> List[Match]:
    from reader.text_detector import YoloDet
    global model
    if model is None:
        model = YoloDet(model_name='yolov8n_table_cell_detect', version=2, score_threshold=0.5)

    field_need_check = ['name', 'birth', 'id', 'address']
    field_name = field_mapping
    if field_name not in field_need_check:
        return last_matches

    image_path = None
    from_page, to_page = 0, 0
    if len(document.pages) > 0:
        image_path = document.pages[0].image_path
        from_page = document.pages[0].page_idx
        to_page = document.pages[-1].page_idx
    report_path = f'{image_path[:-6]}_report_gcn_from{from_page}to{to_page}.json'
    report_path = os.path.join(BASE_DIR, report_path)
    report_file = None

    if os.path.exists(report_path):
        report_file = json.load(open(report_path, 'r'))
        for k, list_matches in report_file.items():
            all_matches = []
            for match_dict in list_matches:
                match = Match()
                match.fromdict(match_dict)
                all_matches.append(match)
            report_file[k] = all_matches
    else:
        '''
            name, brith, id number
        '''
        field_json = {
            'name': {
                'prefix': ['Ông', 'Bà', 'Ông:', 'Bà:'],
                'suffix': ['sinh nam', 'sinh năm', 'năm sinh', 'năm', 'năm sinh :', ',', 'CMND số', 'CMND', 'số'],
                'regex': None
            },
            'birth': {
                'prefix': ['sinh nam', 'sinh năm', 'năm sinh', 'năm', 'năm sinh :', 'Năm :'],
                'suffix': ['CMND số', "CMND", 'số', ',', 'CMND số:'],
                'regex': '[\d ]+'
            },
            'id': {
                'prefix': ['CMND số', "CMND", 'CMND số:'],
                'suffix': ['địa chỉ', 'địa chỉ thường trú', 'Địa chỉ thường trú', 'địa chỉ tại', ',', 'Theo',
                           'địa chỉ thường trú:', 'Địa chỉ thường trú:'],
                'regex': '[\d ]+'
            },
            'address': {
                'prefix': ['địa chỉ', 'địa chỉ thường trú', 'Địa chỉ thường trú', 'địa chỉ tại',
                           'địa chỉ thường trú:', 'Địa chỉ thường trú:'],
                'suffix': ['theo hợp đồng chuyển', 'nhận chuyển nhượng QSD', 'theo hợp đồng', 'theo', "Theo",
                           'Theo nội dung thẩm tra'],
                'regex': None
            }
        }

        rule_priority_custom = [
            RuleNameEnum.BOUNDING_BOX,
            RuleNameEnum.PREFIX,
            RuleNameEnum.SUFFIX,
            RuleNameEnum.REGEX
        ]

        def get_field_from_rule(_document: Document, prefix, suffix, page_idx=None, bounding_box=None, _regex=None,
                                **kwargs):
            document = copy.deepcopy(_document)
            map_match_rule = {}
            last_rule = None
            for rule_name in rule_priority_custom:
                if last_rule is not None:
                    # Tính start_end từ kết quả rule trước
                    last_matches = map_match_rule[last_rule]
                    map_match_rule[rule_name] = []
                    values = []
                    for match in last_matches:
                        # đi theo nhánh match của rule trước đó
                        if rule_name == 'prefix':
                            values = prefix
                        if rule_name == 'suffix':
                            values = suffix

                        matches = rule_mapping[rule_name](match.document, match.text, match.start_idx,
                                                          match.end_idx,
                                                          last_matches=last_matches, last_rule=last_rule,
                                                          values=values, regex=_regex, match=match)
                        map_match_rule[rule_name].extend(matches)
                    # map_match_rule[rule_name] = utils.merge_match(map_match_rule[rule_name])
                    last_rule = rule_name
                else:
                    # lần đầu chạy chưa có rule trước đó | bounding box rule
                    matches = rule_mapping[rule_name](document, document.render(), start_idx, end_idx,
                                                      page_idx=page_idx, bounding_box=bounding_box)
                    matches = utils.merge_match(matches)
                    map_match_rule[rule_name] = matches
                    last_rule = rule_name
            return map_match_rule[last_rule]

        report_dict = {}
        expand = 2

        for idx, page in enumerate(document.pages):
            img = cv2.imread(os.path.join(BASE_DIR, page.image_path))
            table_images = page.tables
            flag_page = False
            for box_table in table_images:
                x1, y1, x2, y2 = box_table
                w, h = x2 - x1, y2 - y1
                crop_table = img[y1: y2, x1: x2]
                # xu ly model luon
                predictions = model([crop_table])
                filter_box = list(filter(lambda x: (x[3][2] - x[3][0]) > w*0.3, predictions[0]))
                for cls, cls_name, pro, box, crop_img in filter_box:
                    x1b, y1b, x2b, y2b = box
                    x1b = max(x1b - expand, 0)
                    y1b = max(y1b - expand, 0)
                    x2b = min(x2b + expand, x2 - x1 + 2)
                    y2b = min(y2b + expand, y2 - y1 + 2)

                    x1b = x1 + x1b
                    y1b = y1 + y1b
                    x2b = x1 + x2b
                    y2b = y1 + y2b
                    # import numpy as np
                    # cv2.rectangle(img, (x1b, y1b), (x2b, y2b), (0, np.random.randint(0, 255), 0))
                    # cv2.imshow('test', img)
                    # cv2.waitKey(0)
                    for name_key, value_key in field_json.items():
                        prefix = value_key['prefix']
                        suffix = value_key['suffix']
                        regex = value_key['regex']
                        list_matches = get_field_from_rule(_document=document, prefix=prefix,
                                                           suffix=suffix, page_idx=idx,
                                                           bounding_box=[x1b, y1b, x2b, y2b], regex=regex)
                        report_dict[name_key] = list_matches
                        if idx == 0 and list_matches is not None and len(list_matches) > 0:
                            flag_page = True
                    '''
                        Flag_page có tác dụng khi mà tại idx = 0 và có giá trị thì nó sẽ out
                    '''
                    if flag_page:
                        break
                if flag_page:
                    break
            if flag_page:
                break
        report_file = copy.deepcopy(report_dict)
        for k, list_matches in report_dict.items():
            report_dict[k] = [match.todict() for match in list_matches]

        with open(report_path, 'w') as file:
            file.write(json.dumps(report_dict))

    return report_file.get(field_mapping)


def custom_rule_chuky():
    pass
