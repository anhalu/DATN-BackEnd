import json
import os.path
import re

import fuzzysearch
import loguru
from content_parser.basic_rule import *
try:
    from content_parser.rule_v6 import check_pattern, get_field_form_page, regex_rule, get_field_with_layout
except:
    from rule_v6 import check_pattern, get_field_form_page, regex_rule, get_field_with_layout
import copy

from content_parser.customer_rule import mapping_cus_rule
import cv2  # for custom rule for read image


def _text_approximate(text: str, query: str, max_dist_per=None, match_case=False):
    if text and query:
        if max_dist_per is None:
            max_dist_per = 10
        if not match_case:
            query = query.lower()
            text = text.lower()
        query = query.strip()
        text = text.replace('\n', ' ').strip()

        # print("Q : ", query)
        # print("T : ", text)

        max_dist = round(len(query) * max_dist_per / 100)
        matches = fuzzysearch.find_near_matches(query, text,
                                                max_l_dist=max_dist)
        if matches:
            matches = list(sorted(matches, key=lambda x: x.dist))
            for match in matches:
                if match.start == 0:
                    return True

    return False


def get_all_word_in_bb(pages, bb, page_idx, list_dimensions):  # bb : x1, y1, x2, y2
    list_word = []
    len_start = 0
    for _idx_page, page in enumerate(pages):
        list_dimensions.append(page['dimensions'])
        for _idx_block, block in enumerate(page['blocks']):
            for _idx_line, line in enumerate(block['lines']):
                for _idx_word, word in enumerate(line['words']):
                    word['start_idx'] = len_start
                    word['end_idx'] = word['start_idx'] + len(word['value'])

                    wv = word['value'].strip()
                    tx, ty = word['bbox'][0]
                    bx, by = word['bbox'][1]
                    mx = (tx + bx) // 2
                    my = (ty + by) // 2

                    if _idx_page == page_idx and (bb[0] <= mx and mx <= bb[2]) and (bb[1] <= my and my <= bb[3]):
                        list_word.append(wv)

                    len_start += len(word['value'])

                    if _idx_word < len(line['words']) - 1:  # space
                        len_start += 1

                if _idx_line < len(block['lines']) - 1:  # \n
                    len_start += 1

            if _idx_block < len(page['blocks']) - 1:  # \n\n
                len_start += 2

        if _idx_page < len(pages) - 1:  # \n\n\n\n
            len_start += 4
            # print("========= listword : ", list_word)
    return ' '.join(list_word)


def get_image_result_yolo(response, pages, list_dimensions):
    for key, value in response.items():
        # key = int(key)
        if len(value['coordinates_title']) > 0:
            value['title'] = get_all_word_in_bb(pages, value['coordinates_title'], int(key), list_dimensions)
        # if len(value['coordinates_number']) > 0: 
        #     value['number'] = get_all_word_in_bb(pages, value['coordinates_number'], int(key), list_dimensions)
    return response


def classify_using_yolo(file: str, rule: dict):
    data_json = json.load(open(file, 'r', encoding='utf8'))
    content = data_json['content']

    map_parent_form = rule['map_parent_form']
    list_map_parent = {i['form_parent']: i['child_parent'] for i in map_parent_form}
    # print(list_map_parent) 
    pages = data_json['pages']
    list_dimensions = []
    response_yolo = {int(key): value for key, value in data_json['response_layout'].items()}
    get_all_word_in_bb(pages, page_idx=0, bb=(0, 0, 0, 0), list_dimensions=list_dimensions)
    response_yolo = get_image_result_yolo(response_yolo, pages, list_dimensions)
    # print("response yolo : ", response_yolo) 
    # END YOLO 

    request_id = rule.get('request_id')
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
    flag_form = False
    start_page = None

    form_match = {}
    for page_idx in range(len(list_page)):
        if response_yolo[page_idx]['status']:
            flag_check = False
            current_form_match = {}
            for form in forms:
                start_pattern = form.get('start_pattern')
                start_pattern_values = start_pattern['values']
                operation = start_pattern['operation']
                max_dist_per = start_pattern['compatible']

                if max_dist_per is None:
                    max_dist_per = 0

                if isinstance(max_dist_per, str):
                    max_dist_per = int(max_dist_per)
                for value in start_pattern_values:
                    check = _text_approximate(response_yolo[page_idx]['title'], query=value, max_dist_per=max_dist_per)
                    if check:
                        flag_check = True
                        current_form_match = form.copy()
                        current_form_match['parent_id'] = form['id']
                        current_form_match['parent_name'] = form['name']
                        break

                if flag_check:
                    break

            if not flag_check:
                current_form_match['id'] = -1
                current_form_match['name'] = "Giấy tờ khác"
                current_form_match['parent_id'] = -1
                current_form_match['parent_name'] = "Giấy tờ khác"

            if flag_form:
                form_response = {}
                form_response['id'] = form_match['id']
                form_response['name'] = form_match['name']
                form_response['parent_id'] = form_match['parent_id']
                form_response['parent_name'] = form_match['parent_name']
                form_response['start_page'] = start_page
                form_response['end_page'] = page_idx - 1
                form_response['field_form'] = []
                form_response['dimensions'] = list_dimensions
                form_response['title'] = response_yolo[start_page]['title']
                form_response['coordinates_title'] = response_yolo[start_page]['coordinates_title']
                form_response[
                    'number'] = f"{response_yolo[start_page]['number']}_{response_yolo[page_idx - 1]['number']}"
                form_response['coordinates_number'] = response_yolo[start_page]['coordinates_number']
                results_response['form_data'].append(form_response)

                start_page = page_idx
                form_match = current_form_match

            else:
                start_page = page_idx
                flag_form = True
                form_match = current_form_match

        if page_idx == len(pages) - 1 and flag_form:
            form_response = {}
            form_response['id'] = form_match['id']
            form_response['name'] = form_match['name']
            form_response['parent_id'] = form_match['parent_id']
            form_response['parent_name'] = form_match['parent_name']
            form_response['start_page'] = start_page
            form_response['end_page'] = page_idx
            form_response['field_form'] = []
            form_response['dimensions'] = list_dimensions
            form_response['title'] = response_yolo[start_page]['title']
            form_response['coordinates_title'] = response_yolo[start_page]['coordinates_title']
            form_response['number'] = f"{response_yolo[start_page]['number']}_{response_yolo[page_idx]['number']}"
            form_response['coordinates_number'] = response_yolo[start_page]['coordinates_number']
            results_response['form_data'].append(form_response)
            flag_form = False

    for form_match in results_response['form_data']:
        parent_name = form_match['parent_name']
        loguru.logger.debug(f'Danh muc form : {parent_name}')
        if form_match['id'] != -1:
            # run 
            form_match['time'] = ""
            form_child_list = list_map_parent[form_match['id']]
            # for child check start_pattern of child form
            form_title = form_match['title']
            flag_status = False
            form_need_check = {}
            len_compatible = -1
            for child_form in form_child_list:
                status, len_compatible_n = check_pattern(form_title, child_form['start_pattern']['values'])
                if status:
                    flag_status = True
                    if len_compatible_n >= len_compatible and len_compatible_n > 0:
                        len_compatible = len_compatible_n
                        form_need_check = copy.deepcopy(child_form)

            if flag_status:
                form_result_get_rule_v6 = get_field_form_page(content, start_list_page[form_match['start_page']],
                                                              end_list_page[form_match['end_page']], form_need_check,
                                                              pages, list_dimensions, end_list_page,
                                                              form_match['start_page'])

                form_match.update(form_result_get_rule_v6)

            """
            apply custom rule, sau nay cho chon hoac lam gi do bla bla
            - tam thoi ap dung cho GCNQSDD
            -   return name, id, birth 
            """
            if form_match['parent_id'] == 1 or form_match['parent_name'] == 'Nhóm giấy chứng nhận':
                start_page = form_match['start_page']
                # from pathlib import Path
                # name_file = os.path.splitext(os.path.basename(file))[0]
                # images = []
                # base_dir = Path(__file__).resolve().parent.parent
                # folder = os.path.join(base_dir, 'data/image/requests')
                # for image_id in range(len(list_page)):
                #     path = name_file + f"_{image_id}.jpg"
                #     path = os.path.join(folder, path)
                #     img = cv2.imread(path)
                #     images.append(img)
                # results = mapping_cus_rule['cus_GCNQSD_v2'](pages=pages, images=images, response_yolo=response_yolo,
                #                                             start_page=form_match['start_page'],
                #                                             end_page=form_match['end_page'],
                #                                             list_dimensions=list_dimensions)
                # form_match['other_fields'] = results

                for field in form_match['field_form']:
                    if field['field_id'] == 1049 or field['name'] == 'Loại GCN':
                        field['value'] = 'Loại A'
                        '''
                        1049 - Loại GCN : Dựa vào số tờ, số thửa trên GCN để xác định loại GCN
                            + Đối với GCN 1 tờ 1 thửa là loại A.
                            + Đối với GCN nhiều tờ nhiều thửa là loại C.
                            + Đối với GCN 1 tờ 1 thửa và có thêm tài sản là loại B.
                        '''

                    if field['field_id'] == 1071 or field['name'] == 'Loại giấy chứng nhận':
                        value = None
                        for field_other in form_match['field_form']:
                            if field_other['name'] == 'Ngày vào sổ':
                                value = field_other['value']
                        if value is not None:
                            regex_value = regex_rule(value, value=r'\d{4}')
                            if len(regex_value) > 0:
                                year = int(regex_value[-1].text)
                                if 1993 <= year <= 2002:
                                    field['value'] = 'Cấp theo luật Đất đai 1993'
                                if 2003 <= year <= 2008:
                                    field['value'] = 'Cấp theo luật Đất đai 2003'
                                if 2009 <= year <= 2013:
                                    field['value'] = 'Cấp theo Nghị định 88'
                                if year >= 2014:
                                    field['value'] = 'Cấp theo Nghị định 43'
                        '''
                        1071 - Loại giấy chứng nhận: Dựa vào ngày ký GCN để xác định loại GCN
                            + Cấp theo luật Đất đai 1993: từ năm 1993 – 2002.
                            + Cấp theo luật Đất đai 2003: Từ năm 2003 -2008.
                            + Cấp theo Nghị định 88: Từ năm 2009 đến năm 2013.
                            + Cấp theo Nghị định 43: Từ năm 2014 trở về đây.
                        '''

                    if field['field_id'] == 1061 or field['name'] == 'Đối tượng sử dụng':
                        first_page = list_page[start_page]
                        value_need_checks = ['CMND', "Năm sinh", "CMND số", 'Địa chỉ thường trú']
                        field_value = 'cá nhân'
                        for value_check in value_need_checks:
                            match = exactly_rule(first_page, value=value_check)
                            if len(match) > 1:
                                field_value = 'hộ gia đình'
                                break
                        field['value'] = field_value
                        '''
                            1061 - Đối tượng sử dụng: Lấy thông tin trên GCN (Dựa vào thông tin chủ tên GCN nếu là để xác định xem nó là hộ gia đình, cá nhân)
                        '''

                    if field['field_id'] == 1078 or field['name'] == 'Ký thay':
                        second_page = list_page[start_page + 1]
                        field_value = 0
                        value_need_checks = ['KT', 'PHÓ CHỦ TỊCH']
                        for value_check in value_need_checks:
                            match = exactly_rule(second_page, value=value_check, max_dist_per=0)
                            if len(match) > 0:
                                field_value = 1
                        field['value'] = str(field_value)
                        '''
                        1078 - Ký thay: Có KT thì để là 1 không có KT thì để là 0
                        '''

                    if field['field_id'] == 1063 or field['name'] == 'Giới tính':
                        field_value = field['value']
                        if "Bà" in field_value:
                            field['value'] = "Nữ"
                        else:
                            field['value'] = "Nam"

        if form_match['id'] == -1 or form_match['id'] == form_match['parent_id']:
            s = form_match['start_page']
            e = form_match['end_page']
            v = '(?i)(Ngày).*(?i)(năm|\/)\s*\d+'
            content_page = list_page[s]
            regex_res = regex_rule(content_page, 0, len(content_page), value=v)
            form_match['time'] = None
            if len(regex_res) > 0:
                form_match['time'] = regex_res[0].text
                start_time = regex_res[0].start_idx + start_list_page[s] - 5
                end_time = regex_res[0].end_idx + start_list_page[s] + 5

                confidence, coordinates, page_idx, dimensions, content_field = get_field_with_layout(pages,
                                                                                                     list_dimensions,
                                                                                                     start_time,
                                                                                                     end_time,
                                                                                                     start_page=s
                                                                                                     )
                if form_match['time'] == '':
                    form_match['time'] = content_field

                if len(coordinates) > 0:
                    form_match['coordinates_time'] = coordinates

    return results_response


if __name__ == '__main__':
    rule = json.load(
        open('new.json', 'r', encoding='utf-8'))

    results_response = classify_using_yolo(
        file="../data/image/requests/a3fd69c9-60fb-4788-9883-e1cfa49a7b8d.json",
        rule=rule)

    print(results_response)
