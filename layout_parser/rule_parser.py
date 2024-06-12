from typing import List


def split_page(boxes: List, threshold=0, width=0, height=0, orientation='vertical'):
    boxes = list(sorted(boxes, key=lambda x: x.min_x if orientation == 'vertical' else x.min_y))
    blocks = [
        {
            'boxes': [boxes[0]],
            'max_x_left': boxes[0].max_x,
            'min_x_right': width,
            'max_y_top': boxes[0].max_y,
            'min_y_bottom': height,
        }
    ]

    for i in range(1, len(boxes)):
        if orientation == 'vertical':
            if boxes[i].min_x <= blocks[-1]['max_x_left']:
                blocks[-1]['boxes'].append(boxes[i])
                blocks[-1]['max_x_left'] = max(blocks[-1]['max_x_left'], boxes[i].max_x)
            else:
                if boxes[i].min_x - boxes[i - 1].max_x > threshold:
                    blocks[-1]['min_x_right'] = boxes[i].min_x
                    blocks[-1]['max_x_left'] = max(boxes[i - 1].max_x, blocks[-1]['max_x_left'])
                    blocks.append({'boxes': [boxes[i]], 'max_x_left': boxes[i].max_x, 'min_x_right': width})
                else:
                    blocks[-1]['max_x_left'] = boxes[i].max_x
                    blocks[-1]['boxes'].append(boxes[i])
        else:
            if boxes[i].min_y <= blocks[-1]['max_y_top']:
                blocks[-1]['boxes'].append(boxes[i])
                blocks[-1]['max_y_top'] = max(blocks[-1]['max_y_top'], boxes[i].max_y)

            else:
                if boxes[i].min_y - boxes[i - 1].max_y > threshold:
                    blocks[-1]['min_y_bottom'] = boxes[i].min_y
                    blocks[-1]['max_y_top'] = max(boxes[i - 1].max_y, blocks[-1]['max_y_top'])
                    blocks.append({'boxes': [boxes[i]], 'max_y_top': boxes[i].max_y, 'min_y_bottom': width})
                else:
                    blocks[-1]['max_y_top'] = boxes[i].max_y
                    blocks[-1]['boxes'].append(boxes[i])
    return [x['boxes'] for x in blocks]


def sort_lines(boxes: List, threshold=0):
    boxes = list(sorted(boxes, key=lambda x: (x.min_y, x.min_x)))
    lines = []
    while boxes:
        b1 = boxes.pop(0)
        line = [b1]

        height_threshold = (b1.min_y + b1.max_y) / 2 + threshold

        for b in boxes:
            if b.min_y < height_threshold:
                line.append(b)

        for b in line:
            if b in boxes:
                boxes.remove(b)
        line = list(sorted(line, key=lambda x: x.min_x))
        lines.append(line)

        boxes = list(sorted(boxes, key=lambda x: (x.min_y, x.min_x)))
    return lines


def parse_layout(page):
    page.blocks = []
    vertical_blocks = split_page(page.lines, threshold=0, width=page.width, height=page.height, orientation='vertical')

    for v in vertical_blocks:
        v_block = Block(x_min=0, y_min=0, x_max=0, y_max=0, name='v_block')
        page.blocks.append(v_block)
        height_line_max = max([x.max_y - x.min_y for x in v])
        page.max_text_height = max(page.max_text_height, height_line_max)
        h = split_page(v, threshold=height_line_max * 1.5, width=page.width, height=page.height,
                       orientation='horizontal')
        for boxes in h:
            h_block = Block(x_min=0, y_min=0, x_max=0, y_max=0, name='h_block')
            v_block.blocks.append(h_block)
            lines = sort_lines(boxes, threshold=0)
            for line in lines:
                min_x = min([b.min_x for b in line])
                min_y = min([b.min_y for b in line])
                max_x = max([b.max_x for b in line])
                max_y = max([b.max_y for b in line])

                l_block = Block(x_min=min_x, y_min=min_y, x_max=max_x, y_max=max_y, name='l_block')
                h_block.blocks.append(l_block)
                l_block.blocks = line
            h_block.x1 = h_block.x4 = min([b.min_x for b in h_block.blocks])
            h_block.y1 = h_block.y2 = min([b.min_y for b in h_block.blocks])
            h_block.x2 = h_block.x3 = max([b.max_x for b in h_block.blocks])
            h_block.y3 = h_block.y4 = max([b.max_y for b in h_block.blocks])

        v_block.x1 = v_block.x4 = min([b.min_x for b in v_block.blocks])
        v_block.y1 = v_block.y2 = min([b.min_y for b in v_block.blocks])
        v_block.x2 = v_block.x3 = max([b.max_x for b in v_block.blocks])
        v_block.y3 = v_block.y4 = max([b.max_y for b in v_block.blocks])
    return page
