import copy

import loguru

from content_parser.basic_rule_doc import *
from application.schemas import *


def form_rule_group_object(_fields: List[FieldValue], form_field_document, main_field_id, list_field_in_object,
                           **kwargs) -> List[FieldValue]:
    """
    :param _fields:
    :param form_field_document:      field document khi ma match duoc
    :param main_field_id:            main field id
    :param list_field_in_object:     gom cac id thanh 1 object
    :param kwargs:
    """
    fields = copy.deepcopy(_fields)
    object_idx = -1
    values_main_field_id = []
    for field, sub_document in zip(fields, form_field_document):
        if field.id == int(main_field_id) and sub_document is not None:
            for value, match_document in zip(field.values, sub_document):
                if match_document is None or value is None:
                    continue
                start_idx = get_content_idx(match_document)
                object_idx += 1
                value.object_idx = object_idx
                values_main_field_id.append([value, object_idx, start_idx])
            break

    if object_idx == -1:  # case not match main_field_id
        return fields

    for field, sub_document in zip(fields, form_field_document):
        if field.id != main_field_id and field.id in list_field_in_object and sub_document is not None:

            '''
                case remove field in field.values if value and coord of value is None and page_idx == -1 ||
                haven't bounding box -> not use keep box in bounding box rule 
                case is working when field.id in list_field_in_obejct
            '''
            field_values = []
            for value in field.values:
                if value.value is not None and value.coord is not None and value.page_idx != -1:
                    field_values.append(value)

            field.values = field_values  # update field values
            for value, match_document in zip(field.values, sub_document):
                if match_document is None or value is None:
                    continue
                start_idx = get_content_idx(match_document)  # get start idx of field
                max_start_idx = None
                check_idx = -1
                for value_main, object_idx_main, start_idx_main in values_main_field_id:
                    if start_idx_main < start_idx and (max_start_idx is None or start_idx_main > max_start_idx):
                        max_start_idx = start_idx_main
                        check_idx = object_idx_main
                value.object_idx = check_idx

    '''
        create new field when field's object_idx is not enough
    '''
    for field in fields:
        if field.id != main_field_id and field.id in list_field_in_object:
            tmp_check = [0] * (object_idx + 1)
            for value in field.values:
                if value.object_idx != -1:
                    tmp_check[value.object_idx] += 1
            for idx, v in enumerate(tmp_check):
                if v == 0:
                    field.values.append(Value(value=None, coord=None, page_idx=-1, object_idx=idx))
    return fields


def get_content_idx(document: Document):
    word_index = None
    for page in document.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word_index is None:
                        word_index = word.word_index
                    else:
                        word_index = min(word_index, word.word_index)
    return word_index


def form_rule_change_field_by_other_field(_fields: List[FieldValue], form_field_document,
                                          target_field,
                                          relation_field,
                                          relation_change,
                                          regex=None,
                                          conversion_to_int=True,
                                          **kwargs):
    """
    :param target_field : field name need to change
    :param relation_field : relation_field
    :param relation_change :
                    relation_change": [
                {
                    "value_check": ["huyện"],
                    "value_return": "0"
                },
                {
                    "value_check": ["tỉnh"],
                    "value_return": "1"
                },
                {
                    "value_check": ["sở"],
                    "value_return": "2"
                }
            ],
    """
    flag_exits = False
    for field in _fields:
        if target_field == field.id or target_field == field.name:
            flag_exits = True
            break
    if not flag_exits:
        return _fields

    fields = copy.deepcopy(_fields)
    values = []
    relation_field_page_idx = None
    relation_field_coord = None
    relation_field_object_idx = None
    for field in fields:
        if relation_field == field.id or relation_field == field.name:
            values = [x.value for x in field.values]
            relation_field_page_idx = field.values[0].page_idx
            relation_field_coord = field.values[0].coord
            relation_field_object_idx = field.values[0].object_idx
            break
    if values[0] is None or len(values) == 0:
        return _fields

    if regex:  # only use for str
        tmp = []
        for value in values:
            matches = re.finditer(regex, value)
            if matches:
                for match in matches:
                    res_tmp = match.group()
                    tmp.append(res_tmp)
        values = tmp

    if conversion_to_int:
        tmp = []
        for value in values:
            try:
                value = int(value)
            except ValueError:
                loguru.logger.error(f"form_rule_change_field_by_other_field : can not convert result to int: {value}")
            tmp.append(value)
        values = tmp

    if len(values) == 0:
        loguru.logger.error("form_rule_change_field_by_other_field : values of relation_field are empty!")
        return _fields

    flag_change = False
    res_target_value = None
    for relation_change_dict in relation_change:
        change_values = relation_change_dict['value_check']
        target_value = relation_change_dict['value_return']
        if isinstance(change_values[0], str):
            checker = 0
            for change_value in change_values:
                for value in values:
                    if change_value in value:
                        checker += 1
                        break
            if checker == len(change_values):  # change value in values
                flag_change = True
        elif isinstance(change_values[0], int) and isinstance(values[0], int):  # only use if values is a list of int
            if len(change_values) == 1:
                checker = sum([1 if value == change_values[0] else 0 for value in values])
                if checker > 0:
                    flag_change = True
            if len(change_values) == 2:
                checker = sum([1 if change_values[0] <= value < change_values[1] else 0 for value in values])
                if checker > 0:
                    flag_change = True
        else:
            loguru.logger.error(
                "form_rule_change_field_by_other_field: not support type of change value in relation change !"
            )

        if flag_change:
            res_target_value = target_value
            break

    if not flag_change:
        return _fields

    for field in fields:
        if target_field == field.id or target_field == field.name:
            field.values = [Value(value=str(res_target_value), page_idx=relation_field_page_idx,
                                  object_idx=relation_field_object_idx, coord=relation_field_coord)]
            break
    return fields
