import re

from content_parser.utils import get_valid_idx_input, _text_approximate, convert_match_value, Match


def exactly_rule(content: str, start_idx=None, end_idx=None, value: str = None, max_dist_per=None, match_case=False):
    start_idx = get_valid_idx_input(start_idx, max_value=len(content), default=0)
    end_idx = get_valid_idx_input(end_idx, max_value=len(content), default=len(content))
    matches = _text_approximate(content[start_idx: end_idx], value, match_case=match_case,
                                max_dist_percent=max_dist_per)
    res = []
    for match in matches:
        res.append(convert_match_value(match, content, start_idx, end_idx))
    return res


def regex_rule(content: str, start_idx=None, end_idx=None, value: str = None, **kwargs):
    start_idx = get_valid_idx_input(start_idx, max_value=len(content), default=0)
    end_idx = get_valid_idx_input(end_idx, max_value=len(content), default=len(content))
    matches = re.finditer(value, content[start_idx: end_idx])
    res = []
    for match in matches:
        m = Match(match.start(), match.end(), None)
        res.append(convert_match_value(m, content, start_idx, end_idx))
    return res
