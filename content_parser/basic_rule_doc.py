import copy
import re
from typing import List, Optional

from application.schemas import RuleNameEnum
from content_parser.utils import text_approximate, convert_match_value, Match, merge_match, sub_doc_from_content_idx
from layout_parser.elements import Document

vocab = ''''aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'''


def exactly_rule(document: Document, content: str,
                 start_idx=None, end_idx=None,
                 value: str = None,
                 max_dist_per=0,
                 match_case=False) -> List[Match]:
    raise Exception()
    matches = text_approximate(text=content,
                               query=value,
                               match_case=match_case,
                               max_dist_percent=max_dist_per)
    res = []
    for match in matches:
        res.append(convert_match_value(match, document.render(), start_idx, end_idx))

    return merge_match(res, intersect=False)


def regex_rule(document: Document, content: str,
               start_idx=None, end_idx=None, regex: str = None,
               max_line: Optional[int] = None, word_exactly=False, _bbox=None, _page_idx=None, **kwargs) -> List[Match]:
    if regex:
        content = document.render()
        matches = re.finditer(regex, content[:])            # content la cai doan match document sau khi render ra (match text) khong can phai start end
        res = []
        for match in matches:
            start_idx = match.start()
            end_idx = match.end()
            if word_exactly:
                if start_idx > 0 and content[start_idx - 1: start_idx] in vocab:
                    continue
                if end_idx < len(content) - 1 and content[end_idx: end_idx + 1] in vocab:
                    continue
            if max_line:
                sub_content = content[start_idx:end_idx]
                lines = sub_content.split('\n')
                idx_line = 0
                end_idx = start_idx
                for line in lines:
                    if len(line) > 0:
                        end_idx += len(line)
                        idx_line += 1
                        if idx_line >= max_line:
                            break
                    end_idx += 1

            m = Match(start_idx=start_idx, end_idx=end_idx, text=content[start_idx:end_idx],
                      document=sub_doc_from_content_idx(document, start_idx, end_idx), bbox=_bbox, page_idx=_page_idx)
            res.append(m)
        return merge_match(res)
    else:
        start_idx = 0
        end_idx = -1
        if max_line:
            sub_content = content[:]
            lines = sub_content.split('\n')
            idx_line = 0
            start_idx = 0
            end_idx = start_idx
            for line in lines:
                if len(line) > 0:
                    end_idx += len(line)
                    idx_line += 1
                    if idx_line >= max_line:
                        break
                end_idx += 1
        if content:
            return [Match(start_idx=start_idx, end_idx=end_idx, text=content[start_idx: end_idx],
                          document=sub_doc_from_content_idx(document, start_idx, end_idx), bbox=_bbox,
                          page_idx=_page_idx)]
        else:
            return [Match(start_idx=start_idx, end_idx=end_idx, text=None,
                          document=document, bbox=_bbox, page_idx=_page_idx)]


def prefix_rule(document: Document, content: str,
                start_idx=None, end_idx=None,
                values: List[str] = None, compatible: int = 0,
                new_line=False, split_line=False, _bbox=None, _page_idx=None, **kwargs) -> List[Match]:
    """
    Get matches after prefix
    :param document:
    :param content:
    :param start_idx:
    :param end_idx:
    :param values:
    :param compatible:
    :param new_line:
    :param split_line:
    :param kwargs:
    :return:
    """
    if values:
        document.update_word_content_index()
        content = document.render()
        pre_matches = []
        for value in values:
            matches = text_approximate(text=document.render(), query=value, max_dist_percent=compatible,
                                       match_case=False)
            pre_matches.extend(matches)

        pre_matches = merge_match(pre_matches, intersect=False)
        all_matches = []
        for match in pre_matches:
            start_idx = match.end_idx + 1
            end_idx = -1
            m = Match(start_idx=start_idx, end_idx=end_idx, text=content[start_idx:end_idx],
                      document=sub_doc_from_content_idx(document, start_idx, end_idx), bbox=_bbox, page_idx=_page_idx)
            all_matches.append(m)
        if all_matches:
            return all_matches
    return [Match(start_idx=0, end_idx=-1, text=content, document=document, bbox=_bbox, page_idx=_page_idx)]


def suffix_rule(document: Document, content: str,
                start_idx=None, end_idx=None, values: List[str] = None, compatible: int = 0,
                new_line=False, split_line=False, last_matches=None, last_rule=None, _bbox=None, _page_idx=None,
                **kwargs) -> List[Match]:
    if values:
        document.update_word_content_index()
        content = document.render()
        pre_matches = []
        for value in values:
            matches = text_approximate(text=document.render(), query=value, max_dist_percent=compatible,
                                       match_case=False)
            pre_matches.extend(matches)

        if last_rule == RuleNameEnum.PREFIX and last_matches and pre_matches:
            min_match = pre_matches[0]
            for match in pre_matches:
                if min_match.start_idx > match.start_idx:
                    min_match = match
            pre_matches = [min_match]

        pre_matches = merge_match(pre_matches, intersect=False, min_end=True)
        all_matches = []
        for match in pre_matches:
            start_idx = 0
            end_idx = match.start_idx
            m = Match(start_idx=start_idx, end_idx=end_idx, text=content[start_idx:end_idx],
                      document=sub_doc_from_content_idx(document, start_idx, end_idx), bbox=_bbox, page_idx=_page_idx)
            all_matches.append(m)
        if all_matches:
            return all_matches
    return [Match(start_idx=0, end_idx=-1, text=content, document=document, bbox=_bbox, page_idx=_page_idx)]


def contain_rule(document: Document, content: str,
                 start_idx=None, end_idx=None, values: str = None, _bbox=None, _page_idx=None, **kwargs) -> List[Match]:
    if values:
        content = document.render()
        return exactly_rule(document, start_idx, end_idx, values, max_dist_per=0, match_case=False)
    return [Match(start_idx=0, end_idx=-1, text=content, document=document, bbox=_bbox, page_idx=_page_idx)]


def page_idx_rule(document: Document,
                  content: str,
                  start_idx=None, end_idx=None, page_idx=0, **kwargs) -> List[Match]:
    """
    Get match of a page in document
    :param document:
    :param content:
    :param start_idx:
    :param end_idx:
    :param page_idx:
    :param kwargs:
    :return:
    """
    base_match = Match(start_idx=start_idx, end_idx=end_idx, text=content, document=document, bbox=None,
                       page_idx=page_idx)

    page = copy.deepcopy(document.pages[page_idx])
    first_word = page.blocks[0].lines[0].words[0]
    last_word = page.blocks[-1].lines[-1].words[-1]
    start_word_idx = first_word.content_index
    end_word_idx = last_word.content_index + len(last_word.value)
    matches = [base_match, Match(start_idx=start_word_idx, end_idx=end_word_idx,
                                 text=page.render(),
                                 document=Document([page]),
                                 bbox=None,
                                 page_idx=page_idx)]
    return merge_match(matches, intersect=True)


def bounding_box_rule(document: Document,
                      content: str,
                      start_idx=None, end_idx=None,
                      page_idx=0, bounding_box=None, keep_box=False, **kwargs) -> List[Match]:
    """
    Từ vị trí trang và bounding box xác định vùng văn bản cần lấy
    :param content: kết quả nội dung của rule trước đó
    :param document:
    :param start_idx:
    :param end_idx:
    :param page_idx:
    :param bounding_box:
    :param kwargs:
    :return:
    """
    if page_idx is None and len(bounding_box) < 4:
        return [Match(start_idx=start_idx, end_idx=end_idx, text=content, document=document)]
    sub_page = document.sub_page_in_bbox(page_idx=page_idx, bbox=bounding_box, contain=True)

    if not keep_box:
        bounding_box = []
        page_idx = None

    if len(bounding_box) == 4 and bounding_box[0] < 1:
        h, w = document.pages[0].dimensions
        bounding_box[0] = int(w * bounding_box[0])
        bounding_box[1] = int(h * bounding_box[1])
        bounding_box[2] = int(w * bounding_box[2])
        bounding_box[3] = int(h * bounding_box[3])

    if len(sub_page.blocks) == 0:
        return [Match(start_idx=None, end_idx=None,
                      text=None, document=Document([sub_page]), bbox=bounding_box, page_idx=page_idx)]

    first_word = sub_page.blocks[0].lines[0].words[0]
    last_word = sub_page.blocks[-1].lines[-1].words[-1]

    start_word_idx = first_word.content_index
    end_word_idx = last_word.content_index + len(last_word.value)

    return [Match(start_idx=start_word_idx, end_idx=end_word_idx,
                  text=sub_page.render(), document=Document([sub_page]), bbox=bounding_box, page_idx=page_idx)]
