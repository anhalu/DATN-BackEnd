from content_parser.rule_v6 import *
from content_parser.utils import _text_approximate

__all__ = ['mapping_cus_rule']

model = None


def cus_GCNQSD_v2(pages, images, response_yolo, start_page, end_page, list_dimensions):
    from reader.text_detector import YoloDet
    global model
    if model is None:
        model = YoloDet(model_name='yolov8n_table_cell_detect', version=2, score_threshold=0.5)

    def get_content_in_page(pages, box, idx):
        res = ""
        list_word = []
        x1_b, y1_b, x2_b, y2_b = box
        for page in pages:
            page_idx = page['page_idx']
            if idx != page_idx:
                continue
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        [x1_w, y1_w], [x2_w, y2_w] = word['bbox']
                        tx = (x1_w + x2_w) / 2
                        ty = (y1_w + y2_w) / 2
                        wv = word['value']
                        if x1_b <= tx <= x2_b and y1_b <= ty <= y2_b:
                            res += (wv + ' ')
                            list_word.append(word)
                            # print(wv)
        # print("END")
        return res, list_word

    def get_field_with_rule(content, prefix_values, suffix_values, max_dist_percent=10):
        start_field = -1
        for v in prefix_values:
            matches = _text_approximate(text=content, query=v, max_dist_percent=max_dist_percent)
            matches = merge_match(matches, intersect=False)

            if matches:
                for match in matches:
                    start_field = max(start_field, match.end_idx)
                    # print(match.start_idx, match.text)
        if start_field == -1:
            return ""
        end_field = -1
        # print("SUFFIX")
        for v in suffix_values:
            matches = _text_approximate(text=content, query=v, max_dist_percent=max_dist_percent)
            matches = merge_match(matches, intersect=False, min_end=True)

            if matches:
                for match in matches:
                    if match.start_idx > start_field:
                        if end_field == -1:
                            end_field = match.end_idx
                        end_field = min(end_field, match.start_idx)
                        # print(match.start_idx, match.text)
        if end_field == -1:
            return ""
        # print("ALL : ", start_field, end_field)
        return content[start_field: end_field]

    table_images = []
    list_table_xyxy = []
    list_table_idx = []
    for (image_idx, value), image in zip(response_yolo.items(), images):
        if start_page <= image_idx <= end_page:
            # print(value['coordinates_table'])
            for (x1, y1, x2, y2) in value['coordinates_table']:
                crop_image = image[y1: y2, x1: x2]
                list_table_xyxy.append([x1, y1, x2, y2])
                list_table_idx.append(image_idx)
                table_images.append(crop_image)
    '''
        name, brith, id number
    '''
    name = birth = id = address = ''
    name_idx = birth_idx = id_idx = address_idx = -1
    coordinates_name = coordinates_birth = coordinates_id = coordinates_address = []
    prefix_name = ['Ông', 'Bà', 'Ông:', 'Bà:']
    suffix_name = ['sinh nam', 'sinh năm', 'năm sinh', 'năm', 'năm sinh :', ',']

    prefix_birth = ['sinh nam', 'sinh năm', 'năm sinh', 'năm', 'năm sinh :', 'Năm :']
    suffix_birth = ['CMND số', "CMND", 'số', ',', 'CMND số:']

    prefix_id = ['CMND số', "CMND", 'CMND số:']
    suffix_id = ['địa chỉ', 'địa chỉ thường trú', 'Địa chỉ thường trú', 'địa chỉ tại', ',', 'Theo',
                 'địa chỉ thường trú:', 'Địa chỉ thường trú:']

    prefix_address = ['địa chỉ', 'địa chỉ thường trú', 'Địa chỉ thường trú', 'địa chỉ tại',
                      'địa chỉ thường trú:', 'Địa chỉ thường trú:']
    suffix_address = ['theo hợp đồng chuyển', 'nhận chuyển nhượng QSD', 'theo hợp đồng', 'theo', "Theo",
                      'Theo nội dung thẩm tra']

    expand = 2
    if len(table_images) > 0:
        for image, table_idx, box_table in zip(table_images, list_table_idx, list_table_xyxy):
            predictions = model([image])
            for cls, cls_name, pro, box, crop_img in predictions[0]:
                x1, y1, x2, y2 = box_table
                x1b, y1b, x2b, y2b = box
                x1b = max(x1b - expand, 0)
                y1b = max(y1b - expand, 0)
                x2b = min(x2b + expand, x2 - x1 + 2)
                y2b = min(y2b + expand, y2 - y1 + 2)
                content, list_word = get_content_in_page(pages, (x1 + x1b, y1 + y1b, x1 + x2b, y1 + y2b), table_idx)

                if ((name == '' and birth == '' and id == '' and address == '')
                        or table_idx - list_table_idx[0] > 1
                        or table_idx > start_page + 1):
                    name = get_field_with_rule(content, prefix_name, suffix_name)
                    birth = get_field_with_rule(content, prefix_birth, suffix_birth)
                    id = get_field_with_rule(content, prefix_id, suffix_id)
                    address = get_field_with_rule(content, prefix_address, suffix_address)
                    coordinates_name = coordinates_birth = coordinates_id = coordinates_address = [x1 + x1b, y1 + y1b,
                                                                                                   x1 + x2b, y1 + y2b]
                    name_idx = birth_idx = id_idx = address_idx = table_idx
                    logger.info(f'Table_idx : {table_idx}')
                    logger.info(f'Content : {content}')

    # regex_name = '[A-Z][a-z]'
    birth_match = regex_rule(content=birth, value='[\d ]+')
    if len(birth_match) > 0:
        birth = birth_match[0].text

    # logger.info(f"?????????? : {id}")
    id_match = regex_rule(content=id, value='[\d ]+')
    if len(id_match) > 0:
        id = id_match[0].text

    results = {
        'name': name,
        'coordinates_name': coordinates_name,
        'name_page_idx': name_idx,
        'birth': birth,
        'coordinates_birth': coordinates_birth,
        'birth_page_idx': birth_idx,
        'id': id,
        'coordinates_id': coordinates_id,
        'id_page_idx': id_idx,
        'address': address,
        'coordinates_address': coordinates_address,
        'address_page_idx': address_idx
    }
    # print(results)
    return results


mapping_cus_rule = {
    'cus_GCNQSD_v2': cus_GCNQSD_v2
}
