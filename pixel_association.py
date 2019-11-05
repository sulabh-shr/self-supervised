import os
import json
import torch
import pickle
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from PIL import Image
import logging
from tqdm import tqdm
from time import time

from defaults import pixel_bbox_assoc, camera, image_defaults, x_resize_factor, \
    y_resize_factor

from active_vision_utils.matlab_utils import load_image_struct, get_tR
from active_vision_utils.projection import camera_to_world_tR, inter_camera_tR, \
    generate_flat_xyz, project_xyz_to_camera, project_camera_to_2d
from active_vision_utils.coordinate_utils import bbox_pixel_indices_list
from active_vision_utils.neighbors import get_hierarchy_neighbors
from active_vision_utils.plot_utils import scatterplot_in_img, bboxplot_in_img

scene_list = ('Home', 'Office')
SAVE_FIG = False
PLOT_FIG = False
# np.random.seed(1)


class PixelBboxAssoc:

    def __init__(self, dataset_root, proposals_root, triplet_root,
                 proposals_per_img=None, hierarchy=None,
                 sample_per_hierarchy=None, proposals_neighbor_img=None,
                 neg_bbox_iou_thresh=None):

        camera_intrinsics = camera['camera_intrinsics']

        if proposals_per_img is None:
            proposals_per_img = pixel_bbox_assoc['proposals_per_img']
        if hierarchy is None:
            hierarchy = pixel_bbox_assoc['hierarchy']
        if sample_per_hierarchy is None:
            sample_per_hierarchy = pixel_bbox_assoc['sample_per_hierarchy']
        if proposals_neighbor_img is None:
            proposals_neighbor_img = pixel_bbox_assoc['proposals_neighbor_img']
        if neg_bbox_iou_thresh is None:
            neg_bbox_iou_thresh = pixel_bbox_assoc['neg_bbox_iou_thresh']

        self.dataset_root = dataset_root
        self.proposals_root = proposals_root
        self.triplet_root = triplet_root
        self.camera_intrinsics = camera_intrinsics
        self.proposals_per_img = proposals_per_img
        self.hierarchy = hierarchy
        self.sample_per_hierarchy = sample_per_hierarchy
        self.proposals_neighbor_img = proposals_neighbor_img
        self.neg_bbox_iou_thresh = neg_bbox_iou_thresh

        self.image_struct_dict = dict()  # k=folder, v={'img_struct', 'scale'}
        self.avd_annotation = dict()
        self.image_folder_map = dict()   # k='img_name', v='img_folder
        self.proposal_filenames = []
        self.proposal_img_filenames = []
        self.neighbors_dict = dict()

        # print(f'Loading Image Parameters and Annotations...')

        # Load image folder map from coco annotation for both train and test
        for j in ['instances_set_1_train.json', 'instances_set_1_test.json']:
            with open(os.path.join(dataset_root, 'coco_annotations', j)) as f:
                self.image_folder_map.update(json.load(f)['img_folder_map'])

        # Load parameters and annotations for each folder in dataset
        for folder in os.listdir(dataset_root):
            folder_path = os.path.join(dataset_root, folder)

            if os.path.isdir(folder_path) and folder.startswith(scene_list):
                img_st, scale = load_image_struct(folder_path)
                self.image_struct_dict[folder] = {'image_struct': img_st,
                                                  'scale': scale}
                ann_path = os.path.join(folder_path, 'annotations.json')
                with open(ann_path, 'r') as f:
                    self.avd_annotation.update(json.load(f))

        # Get neighbors of each proposal img
        # print(f'Creating Neighbors list...')

        for i in os.listdir(proposals_root):

            if i.endswith('.pt'):
                self.proposal_filenames.append(i)
                pt_img_file = self.convert_pt_jpg(i)
                self.proposal_img_filenames.append(pt_img_file)
                self.neighbors_dict[pt_img_file] = \
                    get_hierarchy_neighbors(pt_img_file,
                                            ann=self.avd_annotation,
                                            hierarchy=self.hierarchy)

    def sample_neighbors(self, img_name):
        # import pprint
        neighbors_hierarchy = self.neighbors_dict[img_name]

        # print('Neighbors hierarchy list:')
        # pprint.pprint(neighbors_hierarchy)
        sampled_neighbors = []

        for i, num_sample in enumerate(self.sample_per_hierarchy):
            hierarchy = i + 1
            current_neighbors = neighbors_hierarchy[hierarchy]

            # Choose minimum of number of available neighbors vs sampling size
            sample_size = min(len(current_neighbors), num_sample)
            sampled_neighbors += list(np.random.choice(current_neighbors,
                                                       size=sample_size,
                                                       replace=False))

        return sampled_neighbors

    def load_pt_file(self, proposal_filename):
        try:
            bboxes = torch.load(os.path.join(self.proposals_root,
                                             proposal_filename))[0]
        # Get the bounding boxes
        except FileNotFoundError:
            print(f'File {proposal_filename} not found!!')
            return None

        bboxes_coords = bboxes.bbox.cpu().numpy()
        return bboxes_coords

    def associate(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=os.path.join('logs',
                                                  f'association_{time()}.log'),
                            filemode='w')
        logger.setLevel(logging.DEBUG)

        proposal_filenames = self.proposal_filenames
        proposal_img_filenames = self.proposal_img_filenames

        for proposal_filename, img_filename in tqdm(zip(proposal_filenames,
                                                   proposal_img_filenames),
                                                    total=len(proposal_filenames)):
            # print(f'Fetching samples for {img_filename}')
            logger.info(f'Fetching samples for {img_filename}')
            # print(f'Fetching samples for {img_filename}')

            bboxes = self.load_pt_file(proposal_filename)
            bboxes = bboxes[:self.proposals_per_img]

            img_folder = self.image_folder_map[img_filename]
            img_struct_folder = self.image_struct_dict[img_folder]
            img_struct = img_struct_folder['image_struct']
            img_scale = img_struct_folder['scale']
            tc1w, Rc1w = get_tR(img_filename, img_struct)

            if len(tc1w) == 0:
                logger.debug(f'tR not available for img: {img_filename}')
                print(f'tR not available for img: {img_filename}')
                continue

            tc1w *= img_scale

            twc1, Rwc1 = camera_to_world_tR(tc1w, Rc1w)

            img_path = os.path.join(self.dataset_root, img_folder, 'jpg_rgb',
                                    img_filename)
            img = Image.open(img_path).resize(image_defaults['size'])
            camera_intrinsics = self.camera_intrinsics[img_folder]

            if SAVE_FIG or PLOT_FIG:
                fig, ax = bboxplot_in_img(img, bboxes, fontsize=10, return_fig=True,
                                          linewidth=3)

            depth_name = self.convert_jpg_depth(img_filename)
            depth_path = os.path.join(self.dataset_root, img_folder,
                                      'high_res_depth', depth_name)
            # Scale depth for new image size
            depth = Image.open(depth_path).resize(image_defaults['size'],
                                                  resample=Image.NEAREST)

            x, y, z = generate_flat_xyz(depth)
            cx = camera_intrinsics['cx'] * x_resize_factor
            cy = camera_intrinsics['cy'] * y_resize_factor
            fx = camera_intrinsics['fx'] * x_resize_factor
            fy = camera_intrinsics['fy'] * y_resize_factor

            pcl_cam1, _ = project_xyz_to_camera(x_flat=x, y_flat=y, z_flat=z,
                                                center_x=cx,
                                                center_y=cy, focal_x=fx,
                                                focal_y=fy)
            neighbor_names = self.sample_neighbors(img_filename)
            neighbor_img_path = [
                os.path.join(self.dataset_root, img_folder, 'jpg_rgb',
                             i) for i in neighbor_names]
            neighbor_images = [Image.open(i).resize(image_defaults['size']) for
                               i in neighbor_img_path]

            logger.info(f'Sampled Neighbors : {neighbor_names}')
            # print(f'Sampled Neighbors : {neighbor_names}')

            bboxes_px_idx = bbox_pixel_indices_list(np.array(bboxes), x_flat=x,
                                                    y_flat=y,
                                                    z_flat=z, filter_depth=True,
                                                    coordinates=False)

            if SAVE_FIG or PLOT_FIG:
                for i in bboxes_px_idx:
                    fig, ax = scatterplot_in_img(img, coordinates=(x[i], y[i]), s=1,
                                                 fig=fig, ax=ax, return_fig=True)

                if SAVE_FIG:
                    plt.savefig(os.path.join('/mnt/sda2/workspace/triplet_images',
                                                    img_filename.split('.')[0]+'.png'))
                if PLOT_FIG:
                    plt.show()
                plt.close()
            # {Neighbor_name: [[1st triplet], ...]}
            to_save = OrderedDict()

            for neighbor_name, neighbor_img in zip(neighbor_names,
                                                   neighbor_images):
                to_save[neighbor_name] = []

                # print(f'\nCalculating for neighbor {neighbor_name}')
                logger.info(f'\nCalculating for neighbor {neighbor_name}')
                # print(f'\nCalculating for neighbor {neighbor_name}')

                tc2w, Rc2w = get_tR(neighbor_name, img_struct)

                if len(tc2w) == 0:
                    logger.debug(f'tR not available for neighbor {neighbor_name}')
                    print(f'tR not available for neighbor {neighbor_name}')
                    continue

                tc2w *= img_scale
                tc2c1, Rc2c1 = inter_camera_tR(twc1, Rwc1, tc2w, Rc2w)

                pcl_cam21 = np.matmul(Rc2c1, pcl_cam1) + tc2c1
                proj21 = project_camera_to_2d(pcl_cam21, center_x=cx,
                                              center_y=cy, focal_x=fx,
                                              focal_y=fy)

                neighbor_bboxes = self.load_pt_file(
                    self.convert_jpg_pt(neighbor_name))
                if neighbor_bboxes is None:
                    # print(f'File not found maybe because no proposals in image!!')
                    logger.exception(f'Proposal file: {neighbor_name} not found!!')
                    continue

                neighbor_bboxes = neighbor_bboxes[:self.proposals_neighbor_img]
                # bboxplot_in_img(neighbor_img, neighbor_bboxes)

                # Iterate over each of input image bbox
                for e, bbox_px_idx in enumerate(bboxes_px_idx):
                    # Filter pixels outside image
                    # print(f'Bbox # = {e}')
                    logger.debug(f'Bbox # = {e}')

                    proj_x = proj21[0][bbox_px_idx]
                    proj_y = proj21[1][bbox_px_idx]
                    # scatterplot_in_img(neighbor_img,
                    #                    coordinates=(proj_x, proj_y),
                    #                    s=2)

                    inside_image = np.logical_and.reduce(
                        (proj_x >= 0, proj_x < image_defaults['width'],
                         proj_y >= 0, proj_y < image_defaults['height']))

                    # Only for projected pixels with number of pixels>threshold
                    valid_projected_pxl = np.sum(inside_image)

                    # TODO: Make this a parameter
                    # Skip if number of valid projected pixels is < threshold
                    if valid_projected_pxl < 15:
                        # print(f'Skipping valid pixels = {valid_projected_pxl}')
                        logger.debug(f'Skipping valid pixels = {valid_projected_pxl}')
                        continue

                    proj_x = proj_x[inside_image]
                    proj_y = proj_y[inside_image]

                    # print(f'Num pixels = {valid_projected_pxl}')
                    logger.debug(f'Num pixels = {valid_projected_pxl}')

                    matched_pixels_iou = []

                    # Iterate over each of neighbor image bbox
                    for neighbor_bbox in neighbor_bboxes:
                        area = (neighbor_bbox[2] - neighbor_bbox[0]) * (
                                neighbor_bbox[3] - neighbor_bbox[1])
                        inside_proj_x = np.logical_and(
                            proj_x >= neighbor_bbox[0],
                            proj_x <= neighbor_bbox[2])
                        inside_proj_y = np.logical_and(
                            proj_y >= neighbor_bbox[1],
                            proj_y <= neighbor_bbox[3])
                        num_inside = np.sum(
                            np.logical_and(inside_proj_x, inside_proj_y))
                        # TODO: Note that only using area didn't produce good result because always prioritized smallest box with max pixels inside
                        matched_pixels_iou.append(num_inside / (
                                    area + (valid_projected_pxl - num_inside)))

                    matched_pixels = np.array(matched_pixels_iou)
                    matched_bbox_idx = matched_pixels.argmax()
                    # print(f'IOU = {matched_pixels[matched_bbox_idx]}')
                    logger.debug(f'IOU = {matched_pixels[matched_bbox_idx]}')

                    # TODO: Make this a parameter
                    if matched_pixels[matched_bbox_idx] < 0.2:
                        logger.debug(f'Skipping because small IOU')
                        # print(f'Skipping because small IOU')
                        continue
                    if SAVE_FIG or PLOT_FIG:
                        fig, [ax2, ax1] = plt.subplots(1, 2, figsize=(25, 14))
                        ax1.imshow(neighbor_img)
                        _ = scatterplot_in_img(neighbor_img, [proj_x, proj_y],
                                                     s=2, return_fig=True, fig=fig,
                                               ax=ax1)
                        _ = bboxplot_in_img(neighbor_img,
                                        [neighbor_bboxes[matched_bbox_idx]],
                                        fig=fig, ax=ax1, numbering=False,
                                        return_fig=True, linewidth=3)
                    current_bbox = bboxes[e]

                    found_neg = False

                    # TODO: Make this a parameter
                    max_search_count = min(10, len(bboxes))
                    search_idx_list = list(range(max_search_count))
                    # Shuffle the index list
                    np.random.shuffle(search_idx_list)

                    for neg_bbox_idx in search_idx_list:
                        neg_bbox = bboxes[neg_bbox_idx]

                        intersect_xmin = max(current_bbox[0], neg_bbox[0])
                        intersect_ymin = max(current_bbox[1], neg_bbox[1])
                        intersect_xmax = min(current_bbox[2], neg_bbox[2])
                        intersect_ymax = min(current_bbox[3], neg_bbox[3])
                        intersect_area = (intersect_xmax - intersect_xmin) * (
                                intersect_ymax - intersect_ymin)
                        union_area = (current_bbox[2] - current_bbox[0]) * (
                                current_bbox[3] - current_bbox[1]) + (
                                             neg_bbox[3] - neg_bbox[1]) * (
                                             neg_bbox[2] - neg_bbox[
                                         0]) - intersect_area
                        iou = intersect_area / union_area

                        if iou <= self.neg_bbox_iou_thresh:
                            # print(f'Neg bbox iou = {iou}')
                            logger.debug(f'Neg bbox iou = {iou}')
                            found_neg = True
                            break

                    if not found_neg:
                        logger.debug(f'Did not find negative sample so skipping!!')
                        continue

                    if SAVE_FIG or PLOT_FIG:
                        ax2.imshow(img)
                        _ = bboxplot_in_img(img, [current_bbox, neg_bbox],
                                        fig=fig, ax=ax2, return_fig=True, linewidth=3)
                        if SAVE_FIG:
                            fig_path = os.path.join('/mnt/sda2/workspace/triplet_images',
                                                    img_filename.split('.')[0] + '_' +
                                                    neighbor_name.split('.')[
                                                        0] + '_' + str(e) + '.png'
                                                    )
                            plt.savefig(fig_path)
                        if PLOT_FIG:
                            plt.show()
                        plt.close()
                    to_save[neighbor_name].append(
                        [current_bbox, neighbor_bboxes[matched_bbox_idx],
                         neg_bbox])

            pickle_filename = proposal_filename.split('.')[0] + '.pickle'
            pickle_path = os.path.join(self.triplet_root, pickle_filename)

            with open(pickle_path, 'wb') as f:
                pickle.dump(to_save, f)

    @staticmethod
    def convert_pt_jpg(pt_filename):
        return pt_filename.split('.')[0] + '.jpg'

    @staticmethod
    def convert_jpg_pt(jpg_filename):
        return jpg_filename.split('.')[0] + '.pt'

    @staticmethod
    def convert_jpg_depth(jpg_filename):
        return jpg_filename.split('.')[0][:-1] + '3.png'


def num_px_in_bbox(px, bbox):
    x = px[:, 0]
    y = px[:, 1]
    bbox_xmin = bbox[0]
    bbox_ymin = bbox[1]
    bbox_xmax = bbox[2]
    bbox_ymax = bbox[3]

    x_intersect = np.logical_and(x > bbox_xmin, x < bbox_xmax)
    y_intersect = np.logical_and(y > bbox_ymin, y < bbox_ymax)

    num_inside = np.logical_and(x_intersect, y_intersect)
    return np.sum(num_inside)


if __name__ == '__main__':
    p = PixelBboxAssoc('/mnt/sda2/workspace/DATASETS/ActiveVision',
                       '/home/sulabh/workspace-ubuntu/proposals/av_set1_train_coco',
                       '/home/sulabh/workspace-ubuntu/triplets')
    p.associate()
