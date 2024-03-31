from content_parser.basic_rule_doc import *
from application.schemas import *


def form_rule_group_object(_fields: List[FieldValue], form_field_document, main_field_id, list_field_in_object,
                           **kwargs) -> List[FieldValue]:
    '''
    :param _fields:
    :param form_field_document:      field document khi ma match duoc
    :param main_field_id:            main field id
    :param list_field_in_object:     gom cac id thanh 1 object
    :param kwargs:
    '''
    fields = copy.deepcopy(_fields)
    object_idx = 0
    values_main_field_id = []
    for field, sub_document in zip(fields, form_field_document):
        if field.id == int(main_field_id) and sub_document is not None:
            for value, match_document in zip(field.values, sub_document):
                if match_document is None or value is None:
                    continue
                start_idx = get_content_idx(match_document)
                value.object_idx = object_idx
                values_main_field_id.append([value, object_idx, start_idx])
                object_idx += 1
            break

    for field, sub_document in zip(fields, form_field_document):
        if field.id != main_field_id and field.id in list_field_in_object and sub_document is not None:
            for value, match_document in zip(field.values, sub_document):
                if match_document is None or value is None:
                    continue
                start_idx = get_content_idx(match_document)                         # get start idx of field
                max_start_idx = None
                object_idx = -1
                for value_main, object_idx_main, start_idx_main in values_main_field_id:
                    if start_idx_main < start_idx and (max_start_idx is None or start_idx_main > max_start_idx):
                        max_start_idx = start_idx_main
                        object_idx = object_idx_main
                value.object_idx = object_idx
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
