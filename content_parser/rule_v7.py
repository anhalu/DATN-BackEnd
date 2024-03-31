import glob
import os

from loguru import logger

from content_parser import utils
from content_parser.custom_rule_docs import custom_rule_gcnqsdd, BASE_DIR
from content_parser.form_rule_docs import *
from layout_parser.elements import Document

rule_priority = [RuleNameEnum.PAGE_IDX,
                 RuleNameEnum.BOUNDING_BOX,
                 RuleNameEnum.EXACTLY,
                 RuleNameEnum.PREFIX,
                 RuleNameEnum.SUFFIX,
                 RuleNameEnum.REGEX,
                 RuleNameEnum.CONTAINS]

multi_field_rule_priority = [
    RuleNameEnum.CUSTOM_RULE_GCNQSDD
]

form_rule_priority = [
    RuleNameEnum.FORM_RULE_GROUP_OBJECT
]

rule_mapping = {
    RuleNameEnum.PREFIX: prefix_rule,
    RuleNameEnum.SUFFIX: suffix_rule,
    RuleNameEnum.REGEX: regex_rule,
    RuleNameEnum.CONTAINS: contain_rule,
    RuleNameEnum.EXACTLY: exactly_rule,
    RuleNameEnum.PAGE_IDX: page_idx_rule,
    RuleNameEnum.BOUNDING_BOX: bounding_box_rule,
    RuleNameEnum.CUSTOM_RULE_GCNQSDD: custom_rule_gcnqsdd,
    RuleNameEnum.FORM_RULE_GROUP_OBJECT: form_rule_group_object
}


class GroupFormPages(BaseModel):
    group_form: Optional[GroupForm]
    child_form: Optional[Form]
    title: Optional[str]
    start_page: int
    end_page: int


class ContentParserV7:
    def __init__(self):
        pass

    def _find_group_form(self, document: Document, group_forms: List[GroupForm]) -> List[GroupFormPages]:
        """
        Tìm kiếm form group từ tiêu đề của văn bản
        Nếu chỉ có 1 form group văn bản không có tiêu đề
        thì sẽ lấy luôn form group đó
        :param document:
        :param group_forms:
        :return:
        """
        logger.info(f"{self.__class__.__name__}: find group form {document.info}")
        group_form_pages = []

        # Tại trang đầu tiên thì mở form luôn
        last_start_page = 0
        last_group_form = None
        last_title_form = None
        idx = 0
        for idx, page in enumerate(document.pages):
            title = None
            if page.title_box:
                title = page.title
            if title:
                flag_group = False
                for group in group_forms:
                    # check start pattern if any pattern match with title and start of match = 0 -> open form
                    start_pattern = group.start_pattern
                    start_matches = []
                    for value in start_pattern.values:
                        matches = utils.text_approximate(title, query=value, max_dist_percent=start_pattern.compatible)
                        start_matches.extend([m.start_idx == 0 for m in matches])
                    if any(start_matches):
                        # title khớp với group
                        flag_group = True
                        logger.debug(f"Title: {title} -- match -- {group.name}")
                        if idx > last_start_page:
                            # đóng form cũ
                            child_form = None
                            if last_group_form:
                                child_form = last_group_form.child_forms[0]
                            group_form_pages.append(GroupFormPages(group_form=last_group_form,
                                                                   start_page=last_start_page,
                                                                   end_page=idx - 1,
                                                                   title=last_title_form,
                                                                   child_form=child_form
                                                                   ))
                            # mở form mới
                            last_group_form = group
                            last_start_page = idx
                            last_title_form = title
                        else:
                            # khi chưa có form trước đó thì mở form
                            last_group_form = group
                            last_start_page = idx
                            last_title_form = title
                        break
                if not flag_group:
                    # ko match voi bat ky group nao nhung ma van co title -> GIAY TO KHAC
                    if idx > last_start_page:
                        child_form = None
                        if last_group_form:
                            child_form = last_group_form.child_forms[0]
                        group_form_pages.append(GroupFormPages(group_form=last_group_form,
                                                               start_page=last_start_page,
                                                               end_page=idx - 1,
                                                               title=last_title_form,
                                                               child_form=child_form
                                                               ))
                    # mở form mới
                    last_group_form = None
                    last_start_page = idx
                    last_title_form = title

        if idx >= last_start_page:
            # Nếu có form đang mở thì đóng form (kể cả form trống)
            child_form = None
            if last_group_form:
                child_form = last_group_form.child_forms[0]
            group_form_pages.append(GroupFormPages(group_form=last_group_form,
                                                   start_page=last_start_page,
                                                   end_page=idx,
                                                   title=last_title_form,
                                                   child_form=child_form
                                                   ))
        '''
            Case group forms has 1 form and sub document don't have title
            get the first child_form (value of child form can be None)
        '''
        if len(group_forms) == 1:
            for group_form_page in group_form_pages:
                if group_form_page.title is None:
                    group_form_page.group_form = group_forms[0]
        return group_form_pages

    def _find_form(self, document: Document, title: str, forms: List[Form]) -> Optional[Form]:
        """
        Tìm kiếm form từ tiêu đề của văn bản
        Nếu chỉ có 1 form group, trong form không khai báo start pattern và end pattern và văn bản không có tiêu đề
        thì sẽ lấy luôn form đó
        :param document:
        :param forms:
        :return:
        """
        logger.info(f"{self.__class__.__name__}: find form {document.info}")
        form_matches = []
        for form in forms:
            start_pattern = form.start_pattern
            start_matches = []
            for value in start_pattern.values:
                matches = utils.text_approximate(title, query=value, max_dist_percent=start_pattern.compatible)
                max_match = None
                if matches:
                    max_match = max(matches, key=lambda x: x.length)
                start_matches.append(max_match)
            start_max_match = None
            if start_matches:
                start_max_match = max(start_matches, key=lambda x: 0 if x is None else x.length)
            form_matches.append((form, 0 if start_max_match is None else start_max_match.length))

        form_max_match = max(form_matches, key=lambda x: x[1])

        return form_max_match[0]

    def _find_field_value(self, document: Document, field: Field, return_document=False):
        '''
        document : toàn từ from page -> to page của cả form to
        '''
        logger.debug(f"{self.__class__.__name__}: find {field.name} value")
        input_rules = {r.name: r for r in field.rules}
        if len(field.rules) == 0:
            values = [Value(value=None, coord=None, page_idx=-1)]
            return FieldValue(id=field.id, name=field.name, values=values), [None]
        content = document.render()
        document.update_word_content_index()
        start_idx = 0
        end_idx = len(content)

        map_match_rule = {}
        last_rule = None
        for rule_name in rule_priority:
            if rule_name in input_rules:
                rule = input_rules[rule_name]
                if last_rule is not None:
                    # Tính start_end từ kết quả rule trước
                    last_matches = map_match_rule[last_rule]
                    map_match_rule[rule_name] = []
                    for match in last_matches:
                        # đi theo nhánh match của rule trước đó
                        _bbox = rule.params.get('bounding_box', None) if match.bbox is None else match.bbox
                        _page_idx = rule.params.get('page_idx', None) if match.page_idx is None else match.page_idx
                        matches = rule_mapping[rule.name](match.document, match.text, match.start_idx, match.end_idx,
                                                          last_matches=last_matches, last_rule=last_rule,
                                                          _bbox=_bbox, _page_idx=_page_idx,
                                                          **rule.params)
                        if match.page_idx and match.bbox:
                            for current_match in matches:
                                current_match.get_bbox_page_idx(bbox=match.bbox, page_idx=match.page_idx)
                        map_match_rule[rule_name].extend(matches)
                    # map_match_rule[rule_name] = utils.merge_match(map_match_rule[rule_name])
                    last_rule = rule_name
                else:
                    # lần đầu chạy chưa có rule trước đó
                    matches = rule_mapping[rule.name](document, document.render(), start_idx, end_idx, **rule.params)
                    matches = utils.merge_match(matches)
                    map_match_rule[rule_name] = matches
                    last_rule = rule_name

        last_rule_v2 = None
        map_match_rule_v2 = {}
        for rule_name in multi_field_rule_priority:
            if rule_name in input_rules:
                rule = input_rules[rule_name]
                if last_rule_v2 is not None:
                    # Tính start_end từ kết quả rule trước
                    last_matches = map_match_rule_v2[last_rule_v2]
                    map_match_rule_v2[rule_name] = []
                    for match in last_matches:
                        # đi theo nhánh match của rule trước đó
                        matches = rule_mapping[rule.name](match.document, match.text, match.start_idx, match.end_idx,
                                                          last_matches=last_matches, last_rule=last_rule,
                                                          field_name=field.name,
                                                          **rule.params)
                        if match.page_idx and match.bbox:
                            for current_match in matches:
                                current_match.get_bbox_page_idx(bbox=match.bbox, page_idx=match.page_idx)
                        map_match_rule_v2[rule_name].extend(matches)
                    # map_match_rule[rule_name] = utils.merge_match(map_match_rule[rule_name])
                    last_rule_v2 = rule_name
                else:
                    # lần đầu chạy chưa có rule trước đó
                    matches = rule_mapping[rule.name](document, document.render(), start_idx, end_idx, **rule.params,
                                                      last_matches=map_match_rule[last_rule])
                    matches = utils.merge_match(matches)
                    map_match_rule_v2[rule_name] = matches
                    last_rule_v2 = rule_name

        if last_rule_v2 is not None and map_match_rule_v2[last_rule_v2] is not None and len(map_match_rule_v2[last_rule_v2]) > 0:
            map_match_rule[last_rule] = map_match_rule_v2[last_rule_v2]

        values = []
        list_document = []
        for match in map_match_rule[last_rule]:
            if match.bbox is not None and match.page_idx is not None:
                coord = match.bbox
                page_idx = match.page_idx
            else:
                coord = utils.get_bbox_sub_document(match.document)
                page_idx = utils.find_page_idx_of_value(document, match.document)
            text_value = ""
            if match.text:
                text_value = re.sub(r'\n+', ' ', match.text).strip()
            values.append(Value(value=text_value, coord=coord, page_idx=page_idx))
            list_document.append(match.document)
        field_value = FieldValue(id=field.id, name=field.name, values=values)
        if return_document:
            return field_value, list_document
        return field_value

    def _get_title_value(self, document: Document, start_page_idx: int):
        field_value = FieldValue(id=-1, name="title", values=[
            Value(value=document.pages[0].title, coord=document.pages[0].title_box, page_idx=start_page_idx)
        ])

        return field_value

    def _get_sheet_value(self, document: Document, start_page_idx: int):
        values = []
        for i, page in enumerate(document.pages):
            if page.sheet:
                value = Value(value=page.sheet, coord=page.page_number, page_idx=start_page_idx + i)
                values.append(value)

        field_value = FieldValue(id=-1, name="sheet", values=values)
        return field_value

    def _get_datetime_value(self, document: Document, start_page_idx: int):
        regex_datetime = '(?i)(ngày).{0,25}(?i)(|\/)\s*\d+'
        matches = regex_rule(document, document.render(), None, None, regex_datetime, max_line=1)
        matches = utils.merge_match(matches)
        values = []
        for match in matches:
            coord = utils.get_bbox_sub_document(match.document)
            page_idx = utils.find_page_idx_of_value(document, match.document)
            text_value = ""
            if match.text:
                text_value = re.sub(r'\n+', ' ', match.text).strip()
            values.append(Value(value=text_value, coord=coord, page_idx=page_idx))

        field_value = FieldValue(id=-1, name="datetime", values=values)
        return field_value

    def _parse_form(self, document: Document, form: Optional[Form], from_page: int, to_page: int) -> FormValue:
        dimensions = [document.pages[i].dimensions for i in range(len(document.pages))]
        form_field_document = []
        if form:
            logger.info(f"{self.__class__.__name__}: parse form: {form.name} - {document.info}")
            form_value = FormValue(id=form.id, name=form.name, from_page=from_page, to_page=to_page,
                                   dimensions=dimensions, fields=[])
            for field in form.fields:
                field_value, fields_document = self._find_field_value(document, field, return_document=True)
                form_value.fields.append(field_value)
                form_field_document.append(fields_document)
        else:
            logger.info(f"{self.__class__.__name__}: parse form: Giấy tờ khác - {document.info}")
            form_value = FormValue(id=-1, name="Giấy tờ khác", from_page=from_page, to_page=to_page,
                                   dimensions=dimensions, fields=[])

        # run form rule
        '''
            form rule: đầu vào của nó là các giá trị của field chứ ko phải các 
                document của sub field khi đã chạy xong. 
                còn việc xử lý bên trong sub field là của custom rule for field.  
                
                --> xử lý các field value -> trả ra field value lại 
        '''
        if form is not None and form.form_rules is not None and form_value.fields is not None:
            form_rules = {r.name: r for r in form.form_rules}
            for form_rule_name in form_rule_priority:
                if form_rule_name in form_rules:
                    form_rule = form_rules[form_rule_name]  # get 1 form rule | case multi form rule in one from
                    new_fields = rule_mapping[form_rule.name](_fields=form_value.fields,
                                                              form_field_document=form_field_document,
                                                              **form_rule.params)
                    form_value.fields = []
                    form_value.fields.extend(new_fields)

        form_value.fields.append(self._get_title_value(document, from_page))
        form_value.fields.append(self._get_sheet_value(document, from_page))
        form_value.fields.append(self._get_datetime_value(document, from_page))
        return form_value

    def clear_old_report_data(self, document: Document):
        image_path = document.pages[0].image_path
        report_path = f'{image_path[:-6]}_report_gcn_from*'
        report_path = os.path.join(BASE_DIR, report_path)
        for file in glob.glob(report_path):
            os.remove(file)

    def __call__(self, document: Document,
                 group_forms: List[GroupForm],
                 from_page: Optional[int] = None,
                 to_page: Optional[int] = None, *args, **kwargs) -> List[FormValue]:
        logger.debug(f"Get sub document from page index: {from_page} - {to_page}")

        # sub_document_all = utils.sub_doc_from_page_idx(document, from_page, to_page)

        group_form_pages = self._find_group_form(document, group_forms)
        form_values = []
        for group_form_page in group_form_pages:
            sub_document = utils.sub_doc_from_page_idx(document,
                                                       from_page=group_form_page.start_page,
                                                       to_page=group_form_page.end_page)
            if group_form_page.group_form:
                form = self._find_form(sub_document, title=group_form_page.title,
                                       forms=group_form_page.group_form.child_forms)
                form_value = self._parse_form(sub_document, form=form,
                                              from_page=group_form_page.start_page, to_page=group_form_page.end_page)

            else:
                form_value = self._parse_form(sub_document, form=None,
                                              from_page=group_form_page.start_page, to_page=group_form_page.end_page)

            form_values.append(form_value)
        self.clear_old_report_data(document)
        return form_values


if __name__ == '__main__':
    request_ids = ['7e0b9198-cbce-4b14-a900-09fc7222a886']
    for request_id in request_ids:
        input_body = json.load(
            open('/home/anhalu/anhalu-data/ocr_general_core/content_parser/new.json', 'r', encoding='utf8'))
        request = GroupFormParseRequest(**input_body)
        # request_id = input_body.get('request_id', request_id)
        data = json.load(open(f"../data/image/requests/{request_id}.json", encoding='utf8'))

        document = Document.from_dict(data)
        # print(document.render())

        parser = ContentParserV7()
        a = parser(document, request.group_forms, request.from_page_idx, request.to_page_idx)
        # print(a)
        # print(a[0].json(ensure_ascii=False))
        for b in a:
            print(b.name)
            for f in b.fields:
                print(f'field name: {f.name}: ')
                for v in f.values:
                    print(v.coord)
                    print(f'\t\t{v.object_idx}')
                    print(f'\t\t{v.value[:30]}')
        # json.dump(ensure_ascii=)
