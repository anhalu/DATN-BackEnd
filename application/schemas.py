import json
from enum import Enum
from typing import List, Union, Dict, Optional, Any, Tuple

from pydantic import BaseModel


class FileBase(BaseModel):
    file_id: str
    filename: str
    file_ext: str
    file_size: int
    file_path: str
    status: str
    split_pages: bool


class FileCreate(FileBase):
    pass


class File(FileBase):
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    username: str
    email: str


class UserCreate(UserBase):
    password: str
    master_key: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    id: int
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    is_active: Union[bool, None] = None

    class Config:
        orm_mode = True


class UserInDB(User):
    hashed_password: str
    salt: str


class FieldRule(BaseModel):
    id: int
    name: str
    description: Optional[str]
    type: Optional[str]
    rule: List[Dict]
    page_field: Optional[int] = None
    bounding_box: Optional[List] = None
    keep_bounding_box: Optional[bool] = False


class FormRule(BaseModel):
    id: int
    name: str
    start_pattern: Any
    end_pattern: Any
    description: Optional[str]
    field_form: List[FieldRule]

    def get_field_id(self, field_name):
        for f in self.field_form:
            if f.name == field_name:
                return f.id


class FormParseRequest_classify(BaseModel):
    request_id: str
    form_data: List[FormRule]
    map_parent_form: Optional[List]


class FormParseRequest(BaseModel):
    request_id: str
    form_data: List[FormRule]
    start_check: Optional[int] = None
    end_check: Optional[int] = None
    form_value_id: Optional[int] = None


class FieldRuleResponse(BaseModel):
    field_id: Optional[int]
    name: Optional[str]
    value: Optional[str]
    confidence: Optional[float] = None
    coordinates: Optional[List] = None
    page: Optional[int] = None
    dimensions: Optional[List] = None


class FormRuleResponse(BaseModel):
    id: Optional[int]
    name: Optional[str]
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    field_form: Optional[List[FieldRuleResponse]]
    dimensions: Optional[List]
    title: Optional[str] = None
    form_value_id: Optional[int] = None
    coordinates_title: Optional[List] = None
    number: Optional[str] = None
    coordinates_number: Optional[List]


class FormParseResponse(BaseModel):
    request_id: str
    form_data: Optional[List[FormRuleResponse]] = []


class FormRuleResponseclassify(BaseModel):
    id: Optional[int]
    name: Optional[str]
    parent_id: Optional[int]
    parent_name: Optional[str]
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    field_form: Optional[List[FieldRuleResponse]]
    dimensions: Optional[List]
    title: Optional[str] = None
    coordinates_title: Optional[List] = None
    number: Optional[str] = None
    coordinates_number: Optional[List]
    time: Optional[str] = None
    coordinates_time: Optional[List] = None
    other_fields: Optional[Dict] = {}


class FormParseResponse_classify(BaseModel):
    request_id: str
    form_data: List[FormRuleResponseclassify]


# return list not a str


class FieldRuleResponseVer2(FieldRuleResponse):
    value: Optional[List[str]]
    confidence: Optional[List[float]] = None
    coordinates: Optional[List[List[int]]] = None
    page: Optional[List[int]] = None
    dimensions: Optional[List[List[int]]] = None


class FormRuleResponseVer2(FormRuleResponse):
    field_form: List[FieldRuleResponseVer2]


class FormParseResponseVer2(FormParseResponse):
    form_data: List[FormRuleResponseVer2]


class FormCheckPatternRequest(BaseModel):
    request_id: str
    values: List[str]
    compatible: Optional[int]
    operation: Optional[bool]


class FormCheckPatternResponse(BaseModel):
    filed_id: str
    status: bool
    page_idx: int


class MinioRequest(BaseModel):
    file: str = None
    filename: str = None
    filesize: int = 0


class ExportTxtRequest(BaseModel):
    request_id: str


class FieldTypeEnum(str, Enum):
    STRING = "STRING"
    INT = "INT"
    DATE = "DATE"


class OperationEnum(str, Enum):
    AND = "AND"
    OR = "OR"


class RuleNameEnum(str, Enum):
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CONTAINS = "contains"
    EXACTLY = "exactly"
    REGEX = "regex"
    BOUNDING_BOX = "bounding_box"
    PAGE_IDX = "page_idx"
    CUSTOM_RULE_GCNQSDD = "custom_rule_gcnqsdd"
    CUSTOM_RULE_CHUKY = "custom_rule_chuky"

    # form rule
    FORM_RULE_GROUP_OBJECT = 'form_rule_group_object'


class Rule(BaseModel):
    name: RuleNameEnum
    params: Dict
    priority: int = 1


class Field(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: str
    rules: Optional[List[Rule]] = []


class Value(BaseModel):
    value: Optional[str] = None
    coord: Optional[List[int]] = None
    page_idx: int
    confident: Optional[int] = None
    object_idx: Optional[int] = -1


class FieldValue(BaseModel):
    id: int
    name: str
    values: List[Value]


class FormValue(BaseModel):
    id: int
    name: str
    from_page: int
    to_page: int
    dimensions: List[Tuple[int, int]]
    fields: List[FieldValue]


class DocumentValue(BaseModel):
    request_id: str
    forms: List[FormValue]


class Pattern(BaseModel):
    values: List[str]
    operation: OperationEnum
    compatible: Optional[int] = 20


class FormRuleValue(Rule):
    id: int


class Form(BaseModel):
    id: int
    name: str
    description: str
    start_pattern: Pattern
    end_pattern: Pattern
    fields: List[Field]
    form_rules: Optional[List[FormRuleValue]] = None


class GroupForm(BaseModel):
    id: int
    name: str
    description: str
    start_pattern: Pattern
    end_pattern: Pattern
    child_forms: List[Form]


class GroupFormParseRequest(BaseModel):
    request_id: str
    from_page_idx: Optional[int] = None
    to_page_idx: Optional[int] = None
    group_forms: List[GroupForm]


def convert_old_input_body(input_body: FormParseRequest_classify) -> GroupFormParseRequest:
    request_id = input_body.request_id
    map_parent_form = input_body.map_parent_form
    map_parent_form = {p["form_parent"]: p["child_parent"] for p in map_parent_form}
    parent_form = input_body.form_data
    group_forms = []
    for group_form in parent_form:
        if group_form.id not in map_parent_form:
            continue
        child_forms = map_parent_form[group_form.id]
        new_child_forms = []
        for child_form in child_forms:
            new_fields = []
            for f in child_form['field_form']:
                f['rules'] = f['rule']
                for r in f['rules']:
                    if r['name'] in ['prefix', 'suffix', 'contains']:
                        r['params']['values'] = r['params']['value']
                        r['params']['compatible'] = int(r['params']['compatible'])
                if f['page_field']:
                    f['rules'].append({
                        "name": "page_idx",
                        "priority": 1,
                        "params": {
                            "page_idx": f['page_field']
                        }
                    })
                if f['bounding_box']:
                    f['rules'].append({
                        "name": "bounding_box",
                        "priority": 1,
                        "params": {
                            "page_idx": f['page_field'],
                            "bounding_box": f['bounding_box']
                        }
                    })

                    # print(r)
                new_fields.append(f)

            child_form['start_pattern']['operation'] = 'AND' if child_form['start_pattern']['operation'] else 'OR'
            child_form['end_pattern']['operation'] = 'AND' if child_form['end_pattern']['operation'] else 'OR'
            new_child_form = Form(
                id=child_form['id'],
                name=child_form['name'],
                description=child_form['description'],
                start_pattern=child_form['start_pattern'],
                end_pattern=child_form['end_pattern'],
                fields=new_fields
            )
            new_child_forms.append(new_child_form)

        group_form.start_pattern['operation'] = 'AND' if group_form.start_pattern['operation'] else 'OR'
        group_form.end_pattern['operation'] = 'AND' if group_form.end_pattern['operation'] else 'OR'
        new_group_form = GroupForm(id=group_form.id,
                                   name=group_form.name,
                                   description=group_form.description,
                                   start_pattern=group_form.start_pattern,
                                   end_pattern=group_form.end_pattern,
                                   child_forms=new_child_forms
                                   )
        group_forms.append(new_group_form)
    return GroupFormParseRequest(request_id=request_id,
                                 group_forms=group_forms)


if __name__ == '__main__':
    data = json.load(open('/home/anhalu/anhalu-data/ocr_general_core/content_parser/new.json', encoding='utf8'))
    old_body = FormParseRequest_classify(**data)
    output = convert_old_input_body(old_body)
    # print(type(output))
    print(output.model_dump_json())
    # print(output_json)
    # print(output)
