import numpy as np


def find_boundary_all(axis_a, axis_b, boundary='upper'):
    len_axis_a = len(axis_a)
    len_axis_b = len(axis_b)
    print(axis_a)
    print(axis_b)
    # [[xa1, xa1, ...], [xa2, xa2, ...], ...]
    axis_a_rep = np.repeat(axis_a, len_axis_b).reshape(len_axis_a, len_axis_b)
    print(axis_a_rep)
    # [[xb1, xb2, ...], [xb1, xb2, ...], ...]
    axis_b_tile = np.tile(axis_b, len_axis_a).reshape(len_axis_a, len_axis_b)
    print(axis_b_tile)

    if boundary == 'upper':
        return np.max((axis_a_rep, axis_b_tile), axis=0)
    elif boundary == 'lower':
        return np.min((axis_a_rep, axis_b_tile), axis=0)
    else:
        raise Exception(f'Undefined boundary: {boundary}!!!')


def intersection(bbox1, bbox2):

    iou_x1 = find_boundary_all(bbox1[:, 0], bbox2[:, 0], boundary='upper')
    iou_y1 = find_boundary_all(bbox1[:, 1], bbox2[:, 1], boundary='upper')
    iou_x2 = find_boundary_all(bbox1[:, 2], bbox2[:, 2], boundary='lower')
    iou_y2 = find_boundary_all(bbox1[:, 3], bbox2[:, 3], boundary='lower')

    # iou_y1 = find_boundary_all(bbox1[:, 1], bbox2[:, 1], boundary='upper')
    # iou_x2 = find_boundary_all(bbox1[:, 2], bbox2[:, 2], boundary='lower')
    # iou_y2 = find_boundary_all(bbox1[:, 3], bbox2[:, 3], boundary='lower')


if __name__ == '__main__':
    from plot_utils import bboxplot_in_img

    np.random.seed(5)
    def create_random_bbox():
        bbox = np.random.randint(0, 1080, 4)
        if bbox[0] > bbox[2]:
            bbox[2], bbox[0] = bbox[0], bbox[2]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        return bbox
    n1 = 5
    n2 = 3

    img = np.zeros((1080, 1920, 3), dtype=int) + 240

    bbox_image_1 = np.array([create_random_bbox() for _ in range(n1)])
    bbox_image_2 = np.array([create_random_bbox() for _ in range(n2)])

    # print(bbox_image_1)
    # fig, ax = bboxplot_in_img(img, bbox_image_1, return_fig=True)
    # print(bbox_image_2)
    # bboxplot_in_img(img, bbox_image_2, fig=fig, ax=ax, return_fig=False, edgecolor='r')

    intersection(bbox_image_1, bbox_image_2)
