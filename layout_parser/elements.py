import copy
from pathlib import Path
from typing import *

import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw
from loguru import logger
from matplotlib import pyplot as plt
import cv2
from layout_parser.geometry import resolve_enclosing_rbbox, resolve_enclosing_bbox, BoundingBox, is_box_contained, \
    is_overlap


class Element(object):
    """Implements an abstract document element with exporting and text rendering capabilities"""

    _children_names: List[str] = []
    _exported_keys: List[str] = []

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._children_names:
                setattr(self, k, v)
            else:
                raise KeyError(f"{self.__class__.__name__} object does not have any attribute named '{k}'")

    def export(self) -> Dict[str, Any]:
        """Exports the object into a nested dict format"""

        export_dict = {k: getattr(self, k) for k in self._exported_keys}
        for k, v in export_dict.items():
            if isinstance(v, np.ndarray):
                export_dict[k] = v.tolist()

        for children_name in self._children_names:
            if children_name in ["predictions"]:
                export_dict[children_name] = {
                    k: [item.export() for item in c] for k, c in getattr(self, children_name).items()
                }
            else:
                export_dict[children_name] = [c.export() for c in getattr(self, children_name)]

        return export_dict

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError


class Word(Element):
    """Implements a word element

    Args:
        value: the text string of the word
        confidence: the confidence associated with the text prediction
        bbox: bounding box of the word in format of shape ( 4) - (x1, y1, x2, y2) or (4, 2) - 4 points
         where coordinates are relative to the page's size
    """

    _exported_keys: List[str] = ["value", "confidence", "bbox", "parent_line_id", "word_id"]
    _children_names: List[str] = []
    parent_line_id: str = None

    def __init__(self, value: str, confidence: float, bbox: Union[BoundingBox, np.ndarray],
                 parent_line_id=None, word_id=None) -> None:
        super().__init__()
        self.value = value
        self.confidence = confidence
        self.bbox = bbox
        self.parent_line_id = parent_line_id
        self.word_id = word_id
        self.content_index = 0
        self.word_index = None

    def render(self) -> str:
        """Renders the full text of the element"""
        return self.value

    def extra_repr(self) -> str:
        return f"value='{self.value}', confidence={self.confidence:.2}"

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict.get(k) for k in cls._exported_keys}
        return cls(**kwargs)


class Line(Element):
    """Implements a line element as a collection of words

    Args:
        words: list of word elements
        bbox: bounding box of the word in format of shape (N, 4) - (x1, y1, x2, y2) or (N, 4, 2) - 4 points
            where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all words in it.
    """

    _exported_keys: List[str] = ["bbox"]
    _children_names: List[str] = ["words"]
    words: List[Word] = []

    def __init__(
            self,
            words: List[Word],
            bbox: Optional[Union[BoundingBox, np.ndarray]] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if bbox is None:
            # Check whether this is a rotated or straight box
            box_resolution_fn = resolve_enclosing_rbbox if len(words[0].bbox) == 4 else resolve_enclosing_bbox
            bbox = box_resolution_fn([w.bbox for w in words])  # type: ignore[operator]

        super().__init__(words=words)
        self.bbox = bbox

    def render(self) -> str:
        """Renders the full text of the element"""
        return " ".join(w.render() for w in self.words)

    def update_bbox(self):
        if self.words:
            (x_min, y_min), (x_max, y_max) = self.words[0].bbox
            for word in self.words[1:]:
                (x1, y1), (x2, y2) = word.bbox
                x_min = min(x_min, x1)
                y_min = min(y_min, y1)
                x_max = min(x_max, x2)
                y_max = min(y_max, y1)
            self.bbox = (x_min, y_min), (x_max, y_max)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update(
            {
                "words": [Word.from_dict(_dict) for _dict in save_dict["words"]],
            }
        )
        return cls(**kwargs)


class Block(Element):
    """Implements a block element as a collection of lines and artefacts

    Args:
        lines: list of line elements
        bbox: bounding box of the word in format ((xmin, ymin), (xmax, ymax)) where coordinates are relative to
            the page's size. If not specified, it will be resolved by default to the smallest bounding box enclosing
            all lines and artefacts in it.
    """

    _exported_keys: List[str] = ["bbox"]
    _children_names: List[str] = ["lines"]
    lines: List[Line] = []

    def __init__(
            self,
            lines: List[Line] = [],
            bbox: Optional[Union[BoundingBox, np.ndarray]] = None,
    ) -> None:
        # Resolve the geometry using the smallest enclosing bounding box
        if bbox is None:
            line_boxes = [word.bbox for line in lines for word in line.words]
            box_resolution_fn = (
                resolve_enclosing_rbbox if isinstance(lines[0].bbox, np.ndarray) else resolve_enclosing_bbox
            )
            bbox = box_resolution_fn(line_boxes)  # type: ignore[operator]

        super().__init__(lines=lines)
        self.bbox = bbox

    def render(self, line_break: str = "\n") -> str:
        """Renders the full text of the element"""
        return line_break.join(line.render() for line in self.lines)

    def update_bbox(self):
        if self.lines:
            line = self.lines[0]
            line.update_bbox()
            (x_min, y_min), (x_max, y_max) = line.bbox
            for line in self.lines[1:]:
                line.update_bbox()
                (x1, y1), (x2, y2) = line.bbox
                x_min = min(x_min, x1)
                y_min = min(y_min, y1)
                x_max = min(x_max, x2)
                y_max = min(y_max, y1)
            self.bbox = (x_min, y_min), (x_max, y_max)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update(
            {
                "lines": [Line.from_dict(_dict) for _dict in save_dict["lines"]],
            }
        )
        return cls(**kwargs)


class Page(Element):
    """Implements a page element as a collection of blocks

    Args:
        blocks: list of block elements
        page_idx: the index of the page in the input raw document
        dimensions: the page size in pixels in format (height, width)
        orientation: a dictionary with the value of the rotation angle in degress and confidence of the prediction
        language: a dictionary with the language value and confidence of the prediction
    """

    _exported_keys: List[str] = ["page_idx", "dimensions", "orientation", "language", 'image_path', "figures", "tables",
                                 "title_box",
                                 "page_number", "signature_boxes"]
    _children_names: List[str] = ["blocks"]
    blocks: List[Block] = []

    def __init__(
            self,
            blocks: List[Block],
            page_idx: int,
            dimensions: Tuple[int, int],
            orientation: Optional[Dict[str, Any]] = None,
            language: Optional[Dict[str, Any]] = None,
            image: Optional[np.ndarray] = None,
            image_path: Optional[str] = None,
            figures: List[Tuple[int, int, int, int]] = None,
            tables: List[Tuple[int, int, int, int]] = None,
            title_box: Optional[Tuple[int, int, int, int]] = None,
            page_number: Optional[Tuple[int, int, int, int]] = None,
            signature_boxes: List[Tuple[int, int, int, int]] = None,
    ) -> None:
        super().__init__(blocks=blocks)
        self.page_idx = page_idx
        self.dimensions = dimensions
        self.orientation = orientation if isinstance(orientation, dict) else dict(value=None, confidence=None)
        self.language = language if isinstance(language, dict) else dict(value=None, confidence=None)
        self.image = image
        self.image_path = image_path
        if figures is None:
            figures = []
        self.figures = figures
        if tables is None:
            tables = []
        if signature_boxes is None:
            signature_boxes = []
        self.tables = tables
        self.title_box = title_box
        self.page_number = page_number
        self.signature_boxes = signature_boxes

    # @property
    # def get_page_from_path(self):
    #     if self.image_path:
    #         return cv2.imread(self.image_path)
    #     return []

    @property
    def info(self):
        num_word = 0
        num_line = 0
        num_block = len(self.blocks)
        for block in self.blocks:
            num_line += len(block.lines)
            for line in block.lines:
                num_word += len(line.words)
        return f'Page {self.page_idx} has {num_block} blocks and {num_line} lines and {num_word} words'

    def render(self, block_break: str = "\n\n") -> str:
        """Renders the full text of the element"""
        return block_break.join(b.render() for b in self.blocks)

    def extra_repr(self) -> str:
        return f"dimensions={self.dimensions}"

    def show(self, page: np.ndarray, interactive: bool = True, preserve_aspect_ratio: bool = False, **kwargs) -> None:
        """Overlay the result on a given image

        Args:
            page: image encoded as a numpy array in uint8
            interactive: whether the display should be interactive
            preserve_aspect_ratio: pass True if you passed True to the predictor
        """
        # visualize_page(self.export(), page, interactive=interactive, preserve_aspect_ratio=preserve_aspect_ratio)
        BASE_DIR = Path(__file__).resolve().parent.parent
        fontpath = BASE_DIR / "data_db/fonts/Roboto-Regular.ttf"
        font = ImageFont.truetype(str(fontpath), 15)
        img_pil = Image.fromarray(page)
        draw = ImageDraw.Draw(img_pil)

        if self.title_box:
            x1, y1, x2, y2 = self.title_box
            draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline=(255, 0, 255, 30))
            draw.text((x1, y1), f't', font=font, fill=(0, 0, 255, 255))

        if self.page_number:
            x1, y1, x2, y2 = self.page_number
            draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline=(255, 0, 255, 30))
            draw.text((x1, y1), f'p', font=font, fill=(0, 0, 255, 255))

        for i, f in enumerate(self.figures):
            x1, y1, x2, y2 = f
            draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline=(0, 0, 255, 30))
            draw.text((x1, y1), f'f_{i}', font=font, fill=(0, 0, 255, 255))

        for i, f in enumerate(self.tables):
            x1, y1, x2, y2 = f
            draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline=(255, 0, 255, 30))
            draw.text((x1, y1), f't_{i}', font=font, fill=(0, 0, 255, 255))

        for b_i, block in enumerate(self.blocks):
            for line in block.lines:
                for word in line.words:
                    draw.polygon(word.bbox, fill=(0, 0, 255, 30))
                    orig = word.bbox[0]
                    if word.value:
                        draw.text((orig[0], orig[1] - 10), word.value, font=font, fill=(255, 0, 0, 100))

            (x1, y1), (x2, y2) = block.bbox
            draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], outline=(0, 255, 0, 255))
            draw.text((x1, y1), str(b_i), font=font, fill=(0, 0, 255, 255))
        plt.imshow(img_pil)
        plt.show(**kwargs)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict.get(k) for k in cls._exported_keys}
        kwargs.update({"blocks": [Block.from_dict(block_dict) for block_dict in save_dict["blocks"]]})
        return cls(**kwargs)

    def get_full_lines(self):
        pass

    def remove_text_in_figure(self):
        if self.figures:
            for f in self.figures:
                delete_blocks = []
                for block in self.blocks:
                    delete_lines = []
                    for line in block.lines:
                        delete_words = []
                        for word in line.words:
                            (x1, y1), (x2, y2) = word.bbox
                            if is_box_contained(f, (x1, y1, x2, y2)):
                                delete_words.append(word)
                        if delete_words:
                            for w in delete_words:
                                logger.debug(f"Delete word in figure: {w.bbox} : {w.value}")
                                line.words.remove(w)
                            if len(line.words) == 0:
                                delete_lines.append(line)
                    if delete_lines:
                        for l in delete_lines:
                            block.lines.remove(l)
                        if len(block.lines) == 0:
                            delete_blocks.append(block)
                if delete_blocks:
                    for b in delete_blocks:
                        self.blocks.remove(b)

    def get_text_in_bbox(self, bbox: Tuple[int, int, int, int], contain=True, keep_layout=False):
        """
        Lấy nội dung text trong 1 vùng bbox
        :param bbox: x1, y1, x2, y2
        :param contain: True: chỉ lấy các word nằm hoàn toàn trong bbox, False: lấy cả các word giao với box
        :param keep_layout: True: giữ lại các dấu xuống dòng, False: tất cả các word được join theo dấu cách
        :return:
        """
        blocks = []
        for block in self.blocks:
            lines = []
            for line in block.lines:
                words = []
                for word in line.words:
                    (x1, y1), (x2, y2) = word.bbox
                    if contain:
                        if is_box_contained(bbox, (x1, y1, x2, y2)):
                            words.append(word)
                    else:
                        if is_overlap(bbox, (x1, y1, x2, y2)):
                            words.append(word)
                if words:
                    lines.append(words)
            if lines:
                blocks.append(lines)

        output = ""
        for block in blocks:
            for line in block:
                for word in line:
                    output += word.value + ' '
                output = output.strip()
                if keep_layout:
                    output += '\n'
                else:
                    output += ' '
            output = output.strip()
            if keep_layout:
                output += '\n\n'
            else:
                output += ' '
        return output.strip()

    def sub_page_in_bbox(self, bbox: Tuple[int, int, int, int], contain=True):
        if bbox[0] < 1:
            bbox = copy.deepcopy(bbox)
            bbox[0] *= self.dimensions[1]
            bbox[1] *= self.dimensions[0]
            bbox[2] *= self.dimensions[1]
            bbox[3] *= self.dimensions[0]
        blocks = []
        for block in self.blocks:
            lines = []
            for line in block.lines:
                words = []
                for word in line.words:
                    (x1, y1), (x2, y2) = word.bbox
                    if contain:
                        if is_box_contained(bbox, (x1, y1, x2, y2)):
                            words.append(word)
                    else:
                        if is_overlap(bbox, (x1, y1, x2, y2)):
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
        page = copy.deepcopy(self)
        page.blocks = blocks
        return page

    @property
    def title(self):
        if self.title_box:
            return self.get_text_in_bbox(self.title_box, contain=False, keep_layout=False)
        return None

    @property
    def sheet(self):
        if self.page_number:
            return self.get_text_in_bbox(self.page_number, contain=False, keep_layout=False)
        return None


class Document(Element):
    """Implements a document element as a collection of pages

        Args:
            pages: list of page elements
        """

    _children_names: List[str] = ["pages"]
    pages: List[Page] = []

    def __init__(
            self,
            pages: List[Page]=None,
    ) -> None:
        super().__init__(pages=pages)

    def render(self, page_break: str = "\n\n\n\n") -> str:
        """Renders the full text of the element"""
        return page_break.join(p.render() for p in self.pages)

    def show(self, pages: List[np.ndarray], **kwargs) -> None:
        """Overlay the result on a given image

        Args:
            pages: list of images encoded as numpy arrays in uint8
        """
        for img, result in zip(pages, self.pages):
            result.show(img, **kwargs)

    def remove_text_in_figure(self):
        for page in self.pages:
            page.remove_text_in_figure()

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({"pages": [Page.from_dict(page_dict) for page_dict in save_dict["pages"]]})
        return cls(**kwargs)

    @property
    def info(self):
        pages_info = [page.info for page in self.pages]
        pages_info = '\n\t'.join(pages_info)
        return f'Document has {len(self.pages)} pages: \n\t{pages_info}'

    def update_word_content_index(self, page_break: str = "\n\n\n\n"):
        """
        Tính toán vị trí các từ trên nội dung content được render
        :param page_break: chuối để xác định trang mới
        :return: update trực tiếp vào content_index trong từng word
        """
        idx = 0
        for page in self.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        word.content_index = idx
                        if word.word_index is None:
                            word.word_index = word.content_index
                        idx += len(word.value) + 1  # cộng 1 cho ký tự space
                    idx -= 1  # trừ 1 ký tự space thừa ở cuối dòng
                    idx += 1  # cộng 1 ký tự \n ở cuối dòng

                idx -= 1  # trừ 1 ký tự \n thừa ở cuối block
                idx += 2  # kết thúc block thêm 2 dấu \n
            idx -= 2  # trừ 2 ký tự \n thừa ở cuối page
            idx += len(page_break)

    def get_text_in_bbox(self, page_idx: int, bbox: Tuple[int, int, int, int], contain=True, keep_layout=False):
        if page_idx >= len(self.pages):
            raise Exception(f"Page index {page_idx} out of range. Max value = {len(self.pages) - 1}")
        page = self.pages[page_idx]
        return page.get_text_in_bbox(bbox, contain=contain, keep_layout=keep_layout)

    def sub_page_in_bbox(self, page_idx: int, bbox: Tuple[int, int, int, int], contain=True):
        if page_idx >= len(self.pages):
            raise Exception(f"Page index {page_idx} out of range. Max value = {len(self.pages) - 1}")
        page = self.pages[page_idx]
        return page.sub_page_in_bbox(bbox, contain=contain)
