from collections import defaultdict

pixel_bbox_assoc = {
    'proposals_per_img': 20,
    'sample_per_hierarchy': [4, 2],
    'hierarchy': 2,
    'proposals_neighbor_img': 350,
    'neg_bbox_iou_thresh': 0.1
}

camera = {
    # Camera intrinsics per folder
    'camera_intrinsics': defaultdict(lambda: {'fx': 1.0477637710998533e+03,
                                              'fy': 1.0511749325842486e+03,
                                              'cx': 9.5926120509632392e+02,
                                              'cy': 5.2911546499433564e+02})

}

image_defaults = {
    'org_size': (1920, 1080),
    'org_width': 1920,
    'org_height': 1080,
    'size': (1333, 750),
    'width': 1333,
    'height': 750
}

x_resize_factor = image_defaults['width']/image_defaults['org_width']
y_resize_factor = image_defaults['height']/image_defaults['org_height']

