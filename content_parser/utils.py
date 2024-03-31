import copy
from typing import List, Optional

import fuzzysearch

from layout_parser.elements import Document


class Match:
    def __init__(self, start_idx=None, end_idx=None, text=None, document=None, bbox=None, page_idx=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.text = text
        self.document = document
        self.bbox = bbox                    # only work if keep box is True -> [x1, y1, x2, y2]
        self.page_idx = page_idx            # same

    def get_bbox_page_idx(self, bbox, page_idx):
        self.bbox = bbox
        self.page_idx = page_idx

    @property
    def length(self):
        return self.end_idx - self.start_idx

    def __str__(self):
        return f'Match: {self.start_idx}-{self.end_idx} ({self.end_idx - self.start_idx}): "{self.text}"'

    def todict(self):
        return {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'text': self.text,
            'ducument': self.document.export(),
            'bbox': self.bbox,
            'page_idx': self.page_idx
        }

    def fromdict(self, save_dict):
        self.start_idx = save_dict['start_idx']
        self.end_idx = save_dict['end_idx']
        self.text = save_dict['text']
        self.document = Document().from_dict(save_dict['ducument'])
        self.bbox = save_dict['bbox']
        self.page_idx = save_dict['page_idx']


vocab = ''''aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'''


def _text_approximate(text: str, query: str, max_dist_percent=None, match_case=False) -> List[Match]:
    if text and query:
        if max_dist_percent is None:
            max_dist_percent = 10
        if isinstance(max_dist_percent, str):
            max_dist_percent = int(max_dist_percent)
        if not match_case:
            query = query.lower()
            text = text.lower()
        query = query.strip()
        text = text.replace('\n', ' ')

        max_dist = round(len(query) * max_dist_percent / 100)
        matches = fuzzysearch.find_near_matches(query, text,
                                                max_l_dist=max_dist)
        if matches:
            matches = list(sorted(matches, key=lambda x: x.dist))
            new_matches = []
            for m in matches:
                start = m.start
                end = m.end
                if start > 0 and text[start - 1: start] in vocab:
                    continue
                if end < len(text) - 1 and text[end: end + 1] in vocab:
                    continue
                new_matches.append(Match(start, end, m.matched))
            return new_matches

    return []


def text_approximate(text: str, query: str, max_dist_percent=None, match_case=False) -> List[Match]:
    return _text_approximate(text, query, max_dist_percent, match_case)


def convert_match_value(match, content, start_idx=0, end_idx=-1):
    match_start = start_idx + match.start_idx
    match_end = start_idx + match.end_idx
    text = content[match_start: match_end]
    return Match(match_start, match_end, text)


def get_valid_idx_input(idx, max_value=0, default=0):
    if not idx:
        idx = default
    if isinstance(idx, str):
        idx = int(idx)
    if idx > max_value:
        idx = max_value
    return idx


class NotMatchDocumentException(Exception):
    def __init__(self, doc_name, content, error=None):
        message = f"Document not match with '{doc_name}': {content}"
        if error:
            message = f"Document not match with '{doc_name}' - {error}: {content}"
        super().__init__(message)


def sub_doc_from_page_idx(document: Document, from_page: Optional[int] = None, to_page: Optional[int] = None):
    """

    :param document:
    :param from_page: chỉ số page
    :param to_page:
    :return:
    """
    if from_page is None:
        from_page = 0
    if to_page is None or to_page > len(document.pages):
        to_page = len(document.pages)
    to_page += 1
    sub_doc = Document(document.pages[from_page: to_page])
    return sub_doc


def sub_doc_from_content_idx(document: Document,
                             start_idx: Optional[int] = None,
                             end_idx: Optional[int] = None,
                             contain=True):
    """

    :param document:
    :param start_idx: chỉ số start
    :param end_idx:
    :param contain: True : get all word inner and edge word. False : get only word in index
    :return:
    """
    document.update_word_content_index()
    content = document.render()
    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx > len(content) or end_idx < 0:
        end_idx = len(content)

    pages = []
    for page in document.pages:
        blocks = []
        for block in page.blocks:
            lines = []
            for line in block.lines:
                words = []
                for word in line.words:
                    word_start_idx = word.content_index
                    word_end_idx = word_start_idx + len(word.value)
                    if contain:
                        if start_idx < word_end_idx and end_idx > word_start_idx:
                            words.append(word)
                    else:
                        if start_idx <= word_start_idx and word_end_idx <= end_idx:
                            words.append(word)
                if words:
                    line = copy.deepcopy(line)
                    line.words = words
                    line.update_bbox()
                    lines.append(line)
            if lines:
                block = copy.deepcopy(block)
                block.lines = lines
                block.update_bbox()
                blocks.append(block)
        if blocks:
            page = copy.deepcopy(page)
            page.blocks = blocks
            pages.append(page)

    return Document(pages=pages)


def is_overlap_match(m1: Match, m2: Match):
    s1, e1 = m1.start_idx, m1.end_idx
    s2, e2 = m2.start_idx, m2.end_idx
    return s1 <= s2 < e1 or s2 <= s1 < e2


def merge_match(matches: List[Match], intersect=False, min_end=False) -> List[Match]:
    """
    Merge matches is overlap
    :param matches: list of matches
    :param intersect: True if get only intersect part otherwise False if get union of overlap
    :param min_end: end index is get min of matches overlap
    :return:
    """
    if matches is None or len(matches) == 0:
        return []
    matches = list(sorted(matches, key=lambda x: x.start_idx))
    res = [matches[0]]
    for match in matches[1:]:
        if is_overlap_match(match, res[-1]):
            if min_end:
                res[-1].start_idx = min(match.start_idx, res[-1].start_idx)
                res[-1].end_idx = min(match.end_idx, res[-1].end_idx)
            elif intersect:
                res[-1].start_idx = max(match.start_idx, res[-1].start_idx)
                res[-1].end_idx = min(match.end_idx, res[-1].end_idx)
            else:
                res[-1].start_idx = min(match.start_idx, res[-1].start_idx)
                res[-1].end_idx = max(match.end_idx, res[-1].end_idx)
        else:
            res.append(match)
    return res


def get_bbox_sub_document(document: Document):
    # bbox = []
    # for page in document.pages:
    #     for block in page.blocks:
    #         (x1, y1), (x2, y2) = block.bbox
    #         if not bbox:
    #             bbox = [x1, y1, x2, y2]
    #         else:
    #             bbox = [
    #                 min(bbox[0], x1),
    #                 min(bbox[1], y1),
    #                 max(bbox[2], x2),
    #                 max(bbox[3], y2),
    #             ]
    bbox = []                       # get bbox with word box
    for page in document.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x1, y1), (x2, y2) = word.bbox
                    if not bbox:
                        bbox = [x1, y1, x2, y2]
                    else:
                        bbox = [
                            min(bbox[0], x1),
                            min(bbox[1], y1),
                            max(bbox[2], x2),
                            max(bbox[3], y2)
                        ]
    return bbox


def find_page_idx_of_value(document: Document, sub_document: Document):
    word_ids = set()
    for page in sub_document.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    word_ids.add(word.word_id)
    for page_idx, page in enumerate(document.pages):
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.word_id in word_ids:
                        return page_idx

    return 0
