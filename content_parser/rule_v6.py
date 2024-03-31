import json
import os
import sys
from typing import List

from loguru import logger

from content_parser.basic_rule import exactly_rule, regex_rule
from content_parser.utils import get_valid_idx_input, _text_approximate, NotMatchDocumentException, merge_match

sys.path.append(os.getcwd())


def run_rule(content: str, value: list = [], start_idx=None, end_idx=None, compatible: str = None, **kwargs):
    start_idx = get_valid_idx_input(start_idx, max_value=len(content), default=0)
    end_idx = get_valid_idx_input(end_idx, max_value=len(content), default=len(content))

    if compatible is None:
        compatible = -1

    results = []

    for va in value:
        res = exactly_rule(content=content, start_idx=start_idx, end_idx=end_idx, value=va,
                           max_dist_per=int(compatible))
        # print("value", va, start_idx, end_idx, compatible)  
        # for r in res: 
        #     print(r.start_idx, r.end_idx, r.text) 

        if len(res) > 0:
            [results.append(i) for i in res]
    return results


class Rule:
    def __init__(self, name, priority, params):
        self.name = name
        self.priority = priority
        self.params = params

    @staticmethod
    def from_dict(config):
        rule = Rule(name=config['name'],
                    priority=config['priority'],
                    params=config['params'])

        return rule

    def apply(self, content, start, end):
        n_start = (int(start) + int(self.params['start'])) if (
                self.params['start'] is not None and self.params['start'] != "") else int(start)
        n_end = (int(start) + int(self.params['end'])) if (
                self.params['end'] is not None and self.params['end'] != "") else int(end)
        return run_rule(content=content, start_idx=n_start, end_idx=n_end, **self.params)


def calculate_for_all_box(list_coor):
    xmin, ymin = -1, -1
    xmax, ymax = -1, -1
    for i in list_coor:
        # print(i)
        if xmin == -1:
            xmin, ymin = i[0]
        else:
            xmin = min(xmin, i[0][0])
            ymin = min(ymin, i[0][1])
        if xmax == -1:
            xmax, ymax = i[1]
        else:
            xmax = max(xmax, i[1][0])
            ymax = max(ymax, i[1][1])

    return [xmin, ymin, xmax, ymax]


def narrow_coord(values: List[str], words: List):
    new_coord = []
    for value in values:
        start = -1
        step = 0
        for idx_wb, wb in enumerate(words):
            if wb['value'] == value.split()[0]:
                start = idx_wb
                step = 0
                for idx_w, w in enumerate(value.split()):
                    if w == words[idx_wb + step]['value']:
                        step += 1
                    else:
                        start = -1
                        step = 0
                        break
                if start >= 0:
                    # trường hợp đã bắt được chuỗi
                    new_coord.append(words[start: start + step])

    if len(new_coord) == len(values):
        all_words = []
        for t in new_coord:
            for w in t:
                all_words.append(w)
        new_coord_w = [w['bbox'] for w in all_words]
        new_coord = calculate_for_all_box(new_coord_w)
        return new_coord
    return None


def test_narrow_coord():
    values = ['ngày 09tháng 01 năm 2019']
    words = [{'value': 'Dĩnh', 'confidence': 0.9051045179367065, 'bbox': [[869, 154], [929, 184]], 'start_idx': 97,
              'end_idx': 101},
             {'value': 'Kế,', 'confidence': 0.9106504917144775, 'bbox': [[933, 151], [972, 185]], 'start_idx': 102,
              'end_idx': 105},
             {'value': 'ngày', 'confidence': 0.9006178677082062, 'bbox': [[977, 155], [1035, 188]], 'start_idx': 106,
              'end_idx': 110},
             {'value': '09tháng', 'confidence': 0.8221252858638763, 'bbox': [[1039, 154], [1134, 189]],
              'start_idx': 111,
              'end_idx': 118},
             {'value': '01', 'confidence': 0.8958383798599243, 'bbox': [[1136, 154], [1168, 185]], 'start_idx': 119,
              'end_idx': 121},
             {'value': 'năm', 'confidence': 0.8967201312383016, 'bbox': [[1171, 155], [1223, 185]], 'start_idx': 122,
              'end_idx': 125},
             {'value': '2019', 'confidence': 0.8979768753051758, 'bbox': [[1226, 154], [1286, 185]], 'start_idx': 126,
              'end_idx': 130}]
    # print(narrow_coord(values, words))
    exit(0)


def get_field_with_layout(pages, list_dimensions, start_field, end_field, rule_regex=None, rule_prefix=None, bb=[],
                          page_field=None, start_page=None):
    conf_field = []
    # idx_page = 0 
    list_only_box_for_all_word = []
    list_only_box_for_all_word_all = []
    page_idx = None
    dimensions = None
    flag = True
    res_list = []
    flag_conf = False
    cnt_conf = 0
    if start_page is None:
        start_page = 0
    # print("LAYOUT : ", start_field, end_field)
    for idx_page, page in enumerate(pages):
        if page_field is not None and len(bb) > 0:
            if int(page_field) + start_page < page['page_idx']:
                continue

        # if start_page is not None and page_field is None:
        #     start_page = int(start_page)
        #     if start_page != page['page_idx']:
        #         continue

        for block in page['blocks']:
            for line in block['lines']:

                for word in line['words']:
                    ws = word['start_idx']
                    we = word['end_idx']
                    wv = word['value']
                    tx, ty = word['bbox'][0]
                    bx, by = word['bbox'][1]
                    mx = (tx + bx) // 2
                    my = (ty + by) // 2

                    # if start_field <= ws < end_field and we <= end_field:
                    #     print(ws, mx, my, wv)
                    #     print("???")
                    # if start_field <= ws <= end_field:
                    #     print("???")
                    if start_field <= ws < end_field and we <= end_field:
                        if page_field is not None and len(bb) > 0:
                            if (bb[0] <= mx and mx <= bb[2]) and (bb[1] <= my and my <= bb[3]):
                                flag = False
                                res_list.append(wv.strip())
                                list_only_box_for_all_word.append(word['bbox'])
                                list_only_box_for_all_word_all.append(word)
                                conf_field.append(float(word['confidence']) * 100)
                                if page_idx is None:
                                    page_idx = idx_page
                                dimensions = list_dimensions[idx_page]
                                flag_conf = True
                                # print("???")
                        else:
                            # print("DAMN")
                            flag_conf = True
                            flag = False
                            res_list.append(wv.strip())
                            list_only_box_for_all_word.append(word['bbox'])
                            list_only_box_for_all_word_all.append(word)
                            conf_field.append(float(word['confidence']) * 100)
                            if page_idx is None:
                                page_idx = idx_page
                            dimensions = list_dimensions[idx_page]
                    else:
                        flag_conf = False
                if flag_conf:
                    cnt_conf += 1

                    # idx_page += 1
    confidence = sum(conf_field) / len(conf_field) if len(conf_field) > 0 else 0
    if flag_conf or cnt_conf >= 1:
        confidence = confidence / cnt_conf
    coordinates = calculate_for_all_box(list_only_box_for_all_word)
    content_field = ' '.join(res_list)
    # print(start_field, end_field, content_field, len(res_list))
    # do regex for res
    split_line = None
    if rule_prefix is not None:
        split_line = rule_prefix.params['split_line']

    if rule_regex is not None and rule_regex.params['regex']:
        regex = rule_regex.params['regex']

        if split_line:
            regex = regex[:-2] + '\\n' + regex[-2:]
        content_field = regex_rule(content_field, start_idx=0, end_idx=len(content_field), value=regex)
        try:
            new_coord = narrow_coord([i.text for i in content_field], list_only_box_for_all_word_all)
        except:
            new_coord = None
        if new_coord:
            coordinates = new_coord
        # TODO: update bbox
        if split_line:
            content_field = ' '.join([i.text for i in content_field]).strip().split('\n')
        else:
            content_field = ' '.join([i.text for i in content_field]).strip().replace('\n', '')
    else:
        try:
            new_coord = narrow_coord([i.text for i in content_field], list_only_box_for_all_word_all)
        except:
            new_coord = None
        if new_coord:
            coordinates = new_coord
        if split_line:
            content_field = content_field.strip().split('\n')
        else:
            content_field = content_field.strip().replace('\n', ' ')

    # end regex
    if flag or content_field == "":
        content_field = ""
        page_idx = -1
        coordinates = []
        confidence = -1
        dimensions = []
    return confidence, coordinates, page_idx, dimensions, content_field


def get_start_end_in_bb(pages, bb, page_field, start_page):  # bb : x1, y1, x2, y2
    start_field = None
    end_field = None

    for _idx_page, page in enumerate(pages):
        if page_field is not None and len(bb) > 0:
            if int(page_field) + start_page != page['page_idx']:
                continue
            else:
                h, w = page['dimensions']
                if bb[0] < 1:
                    bb[0] *= w
                    bb[2] *= w
                    bb[1] *= h
                    bb[3] *= h
        for _idx_block, block in enumerate(page['blocks']):
            for _idx_line, line in enumerate(block['lines']):
                for _idx_word, word in enumerate(line['words']):
                    ws = word['start_idx']
                    we = word['end_idx']
                    wv = word['value']
                    tx, ty = word['bbox'][0]
                    bx, by = word['bbox'][1]
                    mx = (tx + bx) // 2
                    my = (ty + by) // 2

                    if (bb[0] <= mx and mx <= bb[2] and bb[1] <= my and my <= bb[3]):
                        if start_field is None:
                            start_field = ws
                        else:
                            start_field = min(ws, start_field)
                        if end_field is None:
                            end_field = we
                        else:
                            end_field = max(we, end_field)

    return start_field, end_field


def get_field_form_page(content, start_in_content, end_in_content, form, pages, list_dimensions, end_list_page,
                        start_page=None):
    logger.debug(f"GO ==== START PAGE : {start_page} NAME FORM ' {form.get('name')} '")
    id_form = form.get('id')
    name_form = form.get('name')
    start_pattern = form.get('start_pattern')
    end_pattern = form.get('end_pattern')
    description_form = form.get('description')
    fields = form.get('field_form')

    form_response = {}
    form_response['id'] = id_form
    form_response['name'] = name_form
    form_response['start_pattern'] = start_pattern
    form_response['end_pattern'] = end_pattern
    form_response['description'] = description_form
    form_response['field_form'] = []

    for field in fields:
        field_response = {}  # init field  response

        bb_field = field.get('bounding_box', None)
        page_field = field.get('page_field', None)
        keep_bounding_box = field.get('keep_bounding_box', False)

        if bb_field is None:
            bb_field = []
            page_field = None

        id_field = field.get('id')
        name_field = field.get('name')

        flag = 0
        rules = field.get('rule')

        start_field = start_in_content
        end_field = None
        if page_field is not None:
            page_end_idx = int(page_field) + int(start_page) if int(page_field) + int(start_page) < len(
                end_list_page) else -1
            end_field = end_list_page[page_end_idx]
            # if have bbouding box, then create start_field, end_field in bouding box
            if len(bb_field) > 0:
                # print("BEFORE : ", bb_field)
                _start_field, _end_field = get_start_end_in_bb(pages, bb_field, page_field, start_page)
                # print("AFTER : , ", bb_field)
                if _start_field is not None:
                    start_field = _start_field
                if _end_field is not None:
                    end_field = _end_field
        else:
            end_field = end_in_content
        # print(start_field, end_field, len(bb_field), bb_field)
        rule_prefix = None
        rule_suffix = None
        rule_regex = None
        rule_contains = None
        flag_prefix_suffix = 0
        if len(rules) > 0:
            rules = [Rule.from_dict(c) for c in rules]
            for rule in rules:
                if rule.name == 'prefix':
                    rule_prefix = rule
                if rule.name == 'suffix':
                    rule_suffix = rule
                if rule.name == 'regex':
                    rule_regex = rule
                if rule.name == 'contains':
                    rule_contains = rule

            if rule_prefix is not None:
                matches_prefix = rule_prefix.apply(content, start_field, end_field)
                matches_prefix = merge_match(matches_prefix, intersect=False)
                if matches_prefix:
                    flag += 1

                    start_current = -1
                    for match in matches_prefix:
                        if start_current == -1:
                            start_current = match.end_idx
                        start_current = min(start_current, match.end_idx)
                    if start_current != -1:
                        start_field = start_current
                elif len(rule_prefix.params['value']) > 0:
                    flag_prefix_suffix = 1
                    # print("start field : ", start_field)

            if rule_suffix is not None:
                matches_suffix = rule_suffix.apply(content, start_field, end_field)
                matches_suffix = merge_match(matches_suffix, intersect=False, min_end=True)
                if matches_suffix:
                    flag += 1
                    end_current = -1
                    for match in matches_suffix:
                        if match.start_idx >= start_field:
                            if end_current == -1:
                                end_current = match.start_idx
                            end_current = min(match.start_idx, end_current)
                    if end_current != -1:
                        end_field = end_current
                # elif len(rule_suffix.params['value']) > 0:
                #     if flag_prefix_suffix == 1:
                #         flag_prefix_suffix = 2

            if rule_prefix is not None and rule_prefix.params['new_line']:
                start_field_n = content.find('\n', start_field, end_field)
                if start_field_n > -1:
                    start_field = start_field_n + 1

            if rule_regex is not None and rule_regex.params['max_line']:
                max_line = int(rule_regex.params['max_line'])
                cnt_line = content[start_field: end_field].count('\n')
                if max_line < cnt_line:
                    end_tmp = start_field
                    while max_line > 0:
                        end_tmp = content.find('\n', end_tmp + 1, end_field)
                        max_line -= 1
                    end_field = end_tmp

            if start_field > end_field:
                logger.error(f"ERROR : {start_field} must less than {end_field}")

                # print(f"S : {start_field}, E : {end_field} ")
        # print(flag_prefix_suffix)
        # print(start_field, end_field)
        # print(f"CONTENT : {content[start_field:end_field]}")
        if flag_prefix_suffix == 0:
            confidence, coordinates, page, dimensions, res = get_field_with_layout(pages, list_dimensions, start_field,
                                                                                   end_field, rule_regex=rule_regex,
                                                                                   rule_prefix=rule_prefix, bb=bb_field,
                                                                                   page_field=(page_field),
                                                                                   start_page=start_page)
        else:
            res = ""
            page = -1
            coordinates = []
            confidence = -1
            dimensions = []

        if rule_contains is not None and len(rule_contains.params['value']) > 0:
            for i in rule_contains['params']['value']:
                run = exactly_rule(res, 0, len(res), i, int(rule_contains['params']['compatible']))
                if len(run) == 0:
                    raise NotMatchDocumentException(name_form, i)

        # print(f"name_field : {name_field}, start : {start_field}, end : {end_field}, res : {res}")
        field_response['field_id'] = id_field
        field_response['name'] = name_field
        field_response['value'] = res
        field_response['confidence'] = confidence
        # keep bb_field when keep_bouding_box = True
        if keep_bounding_box:
            field_response['coordinates'] = bb_field
        else:
            field_response['coordinates'] = coordinates

        field_response['page'] = page
        field_response['dimensions'] = dimensions

        form_response['field_form'].append(field_response)
        form_response['dimensions'] = list_dimensions

    return form_response


def get_all_word_in_bb(pages, bb, page_idx, list_dimensions, flag_n=False):  # bb : x1, y1, x2, y2
    list_word = []
    len_start = 0
    flag_dimension = False
    if len(list_dimensions) == 0:
        flag_dimension = True
    flag_continue = False
    flag_nextline = False
    for _idx_page, page in enumerate(pages):
        if flag_dimension:
            list_dimensions.append(page['dimensions'])
        for _idx_block, block in enumerate(page['blocks']):
            for _idx_line, line in enumerate(block['lines']):
                for _idx_word, word in enumerate(line['words']):
                    word['start_idx'] = len_start
                    word['end_idx'] = word['start_idx'] + len(word['value'])

                    wv = word['value']
                    tx, ty = word['bbox'][0]
                    bx, by = word['bbox'][1]
                    mx = (tx + bx) // 2
                    my = (ty + by) // 2

                    if _idx_page == page_idx and (bb[0] <= mx and mx <= bb[2]) and (bb[1] <= my and my <= bb[3]):
                        flag_continue = True
                        if flag_nextline and flag_n:
                            list_word.append('\n')
                            flag_nextline = False

                        list_word.append(wv)
                    len_start += len(word['value'])

                    if _idx_word < len(line['words']) - 1:  # space
                        len_start += 1

                if _idx_line < len(block['lines']) - 1:  # \n
                    len_start += 1
                if flag_continue:
                    flag_nextline = True
            if _idx_block < len(page['blocks']) - 1:  # \n\n
                len_start += 2

        if _idx_page < len(pages) - 1:  # \n\n\n\n
            len_start += 4
    # logger.info(list_word)
    return ' '.join(list_word)


def get_image_result_yolo(response, pages, list_dimensions):
    for key, value in response.items():
        # key = int(key)
        if len(value['coordinates_title']) > 0:
            value['title'] = get_all_word_in_bb(pages, value['coordinates_title'], int(key), list_dimensions)
        # if len(value['coordinates_number']) > 0: 
        #     value['number'] = get_all_word_in_bb(pages, value['coordinates_number'], int(key), list_dimensions)
    return response


def check_pattern(value_to_check, list_values, operation=False, max_dist_per=20):
    _cnt_match = 0
    len_compatible = 0
    for value in list_values:
        # print(value, value_to_check)
        matches = _text_approximate(value_to_check, value, max_dist_percent=int(max_dist_per))
        if matches:
            _cnt_match += 1
            for match in matches:
                len_compatible = max(len_compatible, len(match.text))
                # print(match.text)
    if operation:
        if _cnt_match >= len(list_values):
            return True, len_compatible
    else:
        if _cnt_match > 0:
            return True, len_compatible
    return False, len_compatible


# chỉ chạy trên 1 form
def parse_docs_using_yolo(file: str, rule: dict):
    data_json = json.load(open(file, 'r', encoding='utf-8'))
    content = data_json['content']

    pages = data_json['pages']
    len_pages = len(pages)
    list_dimensions = []
    response_yolo = {int(key): value for key, value in data_json['response_layout'].items()}
    # run to create word start, end (for sure) 
    get_all_word_in_bb(pages, page_idx=0, bb=(0, 0, 0, 0), list_dimensions=list_dimensions)

    response_yolo = get_image_result_yolo(response_yolo, pages, list_dimensions)
    # print("response yolo : ",response_yolo)
    request_id = rule.get('request_id')
    form_value_id = rule.get('form_value_id')  # moi them vao de tra ra cho a Hung ngay 25 thang 1 nam 2024
    forms = rule.get('form_data')
    results_response = {'request_id': request_id,
                        'form_data': []}
    list_page = content.split('\n\n\n\n')

    ##############################################################################

    start_list_page = []  # list index start of each page 
    end_list_page = []  # list index end of each page
    start_page_current = 0
    end_page_current = len(list_page[0]) - 1
    for i in range(1, len(list_page)):
        start_list_page.append(start_page_current)
        end_list_page.append(end_page_current)

        start_page_current = end_page_current + 4 + 1
        end_page_current = start_page_current + len(list_page[i]) - 1
    start_list_page.append(start_page_current)
    end_list_page.append(end_page_current)

    ############################################################################## 

    start_form_with_page = 0  # start_page
    end_form_with_page = len(list_page) - 1  # end_page

    flag_form = False  # flag check form start
    form_need_check = {}  # form need to check
    flag_end_pattern_exit = False

    start_check = rule.get('start_check')
    end_check = rule.get('end_check')
    if start_check is None:
        start_check = 0
    if end_check is None:
        end_check = len_pages
    else:
        end_check += 1
    end_check = min(end_check, len_pages)
    for page_idx in range(start_check, end_check):
        len_compatible_for_yolo = 0  # sale for compatible yolo
        if not (flag_end_pattern_exit and flag_form):  # check start form with page_idx
            flag_check_pattern = False
            form_get_in_page = {}
            form_get_in_page_without_yolo = {}
            for form in forms:
                start_pattern = form.get('start_pattern')
                start_pattern_values = start_pattern['values']
                operation = start_pattern['operation']

                max_dist_per = start_pattern['compatible']

                if max_dist_per is None:
                    max_dist_per = 10
                    # flag_check_pattern = False

                if response_yolo[page_idx]['status']:
                    operation = False  # default for yolo word exit
                    flag_check_pattern_, len_compatible_current = check_pattern(response_yolo[page_idx]['title'],
                                                                                start_pattern_values, operation,
                                                                                max_dist_per)
                    if flag_check_pattern_ and len_compatible_current > len_compatible_for_yolo:
                        flag_check_pattern = True
                        form_get_in_page = form
                        len_compatible_for_yolo = len_compatible_current

                if not (response_yolo[page_idx]['status']):
                    operation = True
                    flag_check_pattern_, len_compatible_current = check_pattern(list_page[page_idx],
                                                                                start_pattern_values, operation,
                                                                                max_dist_per)
                    if flag_check_pattern_:
                        flag_check_pattern = True
                        form_get_in_page_without_yolo = form

            if len_compatible_for_yolo == 0:
                form_get_in_page = form_get_in_page_without_yolo

            if flag_check_pattern:
                if flag_form and page_idx > 0:  # form open
                    end_form_with_page = page_idx - 1
                    # get field using form
                    form_response = get_field_form_page(content, start_list_page[start_form_with_page],
                                                        end_list_page[end_form_with_page], form_need_check, pages,
                                                        list_dimensions, end_list_page,
                                                        start_page=start_form_with_page)  # get field

                    form_response['title'] = response_yolo[start_form_with_page]['title']
                    form_response['coordinates_title'] = response_yolo[start_form_with_page]['coordinates_title']
                    form_response['number'] = response_yolo[start_form_with_page]['number']
                    form_response['coordinates_number'] = response_yolo[start_form_with_page]['coordinates_number']

                    form_response['form_value_id'] = form_value_id  # new update

                    form_response['start_page'] = start_form_with_page
                    form_response['end_page'] = end_form_with_page
                    results_response['form_data'].append(form_response)
                    # new form -> start  
                    flag_form = True
                    start_form_with_page = page_idx
                    end_form_with_page = end_check - 1
                    form_need_check = form_get_in_page
                else:
                    flag_form = True
                    start_form_with_page = page_idx
                    form_need_check = form_get_in_page
                if len(form_need_check.get('end_pattern')['values']) > 0:
                    flag_end_pattern_exit = True

                    # not found in yolo, then run with out yolo

        if flag_form and flag_end_pattern_exit:  # form open and end_pattern exit
            end_pattern = form_need_check.get('end_pattern')
            end_pattern_values = end_pattern['values']
            operation = end_pattern['operation']
            max_dist_per = end_pattern['compatible']

            flag_check_pattern, len_compatible_current = check_pattern(list_page[page_idx], end_pattern_values,
                                                                       operation, max_dist_per=int(max_dist_per))

            if flag_check_pattern:
                end_form_with_page = page_idx
                flag_form = False

                form_response = get_field_form_page(content, start_list_page[start_form_with_page],
                                                    end_list_page[end_form_with_page], form_need_check, pages,
                                                    list_dimensions, end_list_page,
                                                    start_page=start_form_with_page)  # get field
                form_response['title'] = response_yolo[start_form_with_page]['title']
                form_response['coordinates_title'] = response_yolo[start_form_with_page]['coordinates_title']
                form_response['number'] = response_yolo[start_form_with_page]['number']
                form_response['coordinates_number'] = response_yolo[start_form_with_page]['coordinates_number']

                form_response['form_value_id'] = form_value_id  # new update

                form_response['start_page'] = start_form_with_page
                form_response['end_page'] = end_form_with_page
                results_response['form_data'].append(form_response)
                start_form_with_page = page_idx + 1
                end_form_with_page = end_check - 1

        if flag_form and page_idx == end_check - 1:
            end_form_with_page = end_check - 1
            form_response = get_field_form_page(content, start_list_page[start_form_with_page],
                                                end_list_page[end_form_with_page], form_need_check, pages,
                                                list_dimensions, end_list_page,
                                                start_page=start_form_with_page)  # get field
            form_response['title'] = response_yolo[start_form_with_page]['title']
            form_response['coordinates_title'] = response_yolo[start_form_with_page]['coordinates_title']
            form_response['number'] = response_yolo[start_form_with_page]['number']
            form_response['coordinates_number'] = response_yolo[start_form_with_page]['coordinates_number']

            form_response['form_value_id'] = form_value_id  # new update

            form_response['start_page'] = start_form_with_page
            form_response['end_page'] = end_form_with_page

            flag_form = False
            results_response['form_data'].append(form_response)

    list_sort = results_response['form_data']
    for i in range(len(list_sort)):
        list_sort[i]['field_form'] = sorted(list_sort[i]['field_form'], key=lambda x: x['field_id'], reverse=False)

    '''
        FIX GIA TRI CHO FIELD THEO YEU CAU. 
        id_from = 200 : so xay dung hai phong 
        id_field = 765 : id field chu ky
    '''
    for form in list_sort:
        if form['id'] == 200:  # SO XAY DUNG HAI PHONG
            start_page = form['start_page']
            end_page = form['end_page']
            flag_break = False
            for field in form['field_form']:
                if field['field_id'] == 765:
                    # lay ra signature
                    for img_idx in range(start_page, end_page + 1):
                        h, w = list_dimensions[img_idx]
                        coordinates_signature = response_yolo[img_idx]['coordinates_signature']
                        if len(coordinates_signature) > 0:
                            best_coor = [0, 0, 0, 0]
                            for coord in coordinates_signature:
                                x1, y1, x2, y2 = coord
                                if (y2 - y1) > (best_coor[3] - best_coor[1]):
                                    best_coor = coord
                            if best_coor != [0, 0, 0, 0] and (best_coor[3] - best_coor[1]) > w * 5 / 100:
                                field['value'] = get_all_word_in_bb(pages, best_coor, img_idx, list_dimensions, True).split('\n')[-1]
                                field['coordinates'] = best_coor
                                field['page'] = img_idx
                                field['dimensions'] = [h, w]
                                flag_break = True
                                break
                        if flag_break:
                            break

                    if not flag_break:
                        field['value'] = ""
                        field['coordinates'] = []
                        field['page'] = -1
                        field['dimensions'] = []

                if flag_break:
                    break

            logger.info(field)
    results_response['form_data'] = list_sort

    return results_response


if __name__ == '__main__':
    rule = json.load(
        open('/home/anhalu/anhalu-data/ocr_general_core/content_parser/all_form.json', 'r', encoding='utf-8'))

    form_response = parse_docs_using_yolo(
        file="/home/anhalu/anhalu-data/ocr_general_core/data/image/requests/9e89ba31-32ea-4394-9f6d-e51a9f91d4f0.json",
        rule=rule)
    print(form_response)

    # for _form in form_response['form_data']: 
    #     print("============", _form)
