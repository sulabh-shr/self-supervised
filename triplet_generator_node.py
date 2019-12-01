import os
import json
import pickle
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

import tqdm
import logging
from datetime import datetime

from active_vision_utils.neighbors import distance_sort_nodes
from active_vision_utils.matlab_utils import load_image_struct, get_tR, get_image_from_nodes
from active_vision_utils.projection import camera_to_world_tR, generate_flat_xyz, \
    project_xyz_to_camera, inter_camera_tR, project_camera_to_2d
from active_vision_utils.coordinate_utils import bbox_pixel_indices_list
from active_vision_utils.plot_utils import bboxplot_in_img, scatterplot_in_img

from triplet_generator import TripletGenerator


class TripletGeneratorNode(TripletGenerator):

    def __init__(self, dataset_path, proposals_path, triplet_save_path, scene, camera_intrinsics,
                 sampling_params, image_params, match_params, plot_params):
        super(TripletGeneratorNode, self).__init__(dataset_path=dataset_path,
                                                   proposals_path=proposals_path,
                                                   triplet_save_path=triplet_save_path)

        self.scene = scene
        self.camera_intrinsics = camera_intrinsics
        self.sampling_params = sampling_params
        self.image_params = image_params
        self.match_params = match_params
        self.plot_params = plot_params

        # SET PATHS
        self.scene_path = os.path.join(dataset_path, scene)
        self.img_folder_path = os.path.join(self.scene_path, 'jpg_rgb')
        self.depth_folder_path = os.path.join(self.scene_path, 'high_res_depth')

        # LOAD ANNOTATIONS AND MATLAB FILES
        with open(os.path.join(self.scene_path, 'annotations.json')) as f:
            self.annotation = json.load(f)
        self.image_struct, self.scale = load_image_struct(self.scene_path)

        # RESCALE TO PROPOSAL IMAGE SIZE
        self.cx_prop = self.camera_intrinsics['cx'] * self.image_params['x_scale_org_to_proposal']
        self.cy_prop = self.camera_intrinsics['cy'] * self.image_params['y_scale_org_to_proposal']
        self.fx_prop = self.camera_intrinsics['fx'] * self.image_params['x_scale_org_to_proposal']
        self.fy_prop = self.camera_intrinsics['fy'] * self.image_params['y_scale_org_to_proposal']

        # MAKE FOLDERS FOR SAVING
        triplet_train_test_folder = ['Train', 'Test']

        for set_folder in triplet_train_test_folder:
            triplet_set_path = os.path.join(self.triplet_save_path, set_folder)

            if os.path.exists(triplet_set_path):
                if len(os.listdir(triplet_set_path)) != 0:
                    print(f'Path {triplet_set_path} already exists!')
                    raise Exception
            else:
                os.mkdir(triplet_set_path)
        self.plot_save_path = os.path.join(self.triplet_save_path, 'plots')

        if os.path.exists(self.plot_save_path):
            if len(os.listdir(self.plot_save_path)) != 0:
                print(f'Path {self.plot_save_path} already exists!')
                raise Exception
        else:
            os.mkdir(self.plot_save_path)

        # LOGGING
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
        logging.basicConfig(filename=os.path.join('logs',
                                                  f'triplet_gen_{self.start_time}.log'),
                            filemode='w')
        self.logger.setLevel(logging.DEBUG)

    def generate_train_test(self):
        cluster_centers, cluster_nodes = distance_sort_nodes(image_struct=self.image_struct,
                                                             scale=self.scale, near_threshold=25,
                                                             visualize_nodes=False)
        num_nodes = len(cluster_centers)
        num_train = round(num_nodes * 0.75)

        self.train_cluster_centers = cluster_centers[:num_train]
        self.train_clusters = cluster_nodes[:num_train]
        self.test_cluster_centers = cluster_centers[num_train:]
        self.test_clusters = cluster_nodes[num_train:]

        if self.plot_params['clusters'] or self.plot_params['save_clusters']:
            set_cluster = ['Train', 'Test']

            for set_idx, clusters in enumerate([self.train_cluster_centers, self.test_cluster_centers]):
                plt.figure(figsize=(12, 8))
                plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1])

                if set_idx == 0:
                    plt.plot(clusters[0, 0], clusters[0, 1], 'ro', label='Starting Node')

                for i in range(len(clusters)):
                    plt.scatter(clusters[i][0], clusters[i][1], edgecolors='b', c='white')

                plt.title(f'Center nodes of the cluster for {set_cluster[set_idx]}')
                plt.axis('equal')
                plt.legend()

                if self.plot_params['clusters']:
                    plt.show()

                if self.plot_params['save_clusters']:
                    plt.savefig(os.path.join(self.plot_save_path,
                                             f'clusters_{set_cluster[set_idx]}.png'),
                                dpi=300)
                plt.close()

    def sample_ref_views(self):
        """
            Iterate over train and test clusters
                Create a list of all views in train all cluster
                Iterate over each cluster
                    Create a shuffled list of available indices of views
                        Iterate of each index
                            Add to sample list if it's FORWARD is also in all list of all views
                                If number of samples met, break
        """

        selected_views_train_test = []

        for clusters in [self.train_clusters, self.test_clusters]:

            views_per_cluster = []
            all_views = []

            for cluster in clusters:
                current_cluster_views = get_image_from_nodes(nodes=cluster, image_struct=self.image_struct)
                views_per_cluster.append(current_cluster_views)
                all_views += current_cluster_views

            selected_views = []

            for cluster_views in views_per_cluster:

                num_views_in_cluster = len(cluster_views)
                available_indices = list(range(num_views_in_cluster))
                available_indices_shuffled = np.random.choice(available_indices,
                                                              size=num_views_in_cluster,
                                                              replace=False)

                num_sampled = 0

                for random_idx in available_indices_shuffled:
                    view_name = cluster_views[random_idx]

                    if self.annotation[view_name]["forward"] in all_views:
                        selected_views.append(view_name)
                        num_sampled += 1

                    if num_sampled == self.sampling_params['sample_per_cluster']:
                        break
            selected_views_train_test.append(selected_views)

        self.train_ref_views = selected_views_train_test[0]
        self.test_ref_views = selected_views_train_test[1]

    def generate_neighbors(self, view):
        """
            Initialize hierarchy 0 to input view
            For depth 1 to max
                For view in current depth views
                    Create a list of views already in the neighbors dict
                    For direction in directions to take
                        Get the view in this direction
                        If this view is not empty and not already counted, add to dict
            Remove input view 0 from dict
        """
        ann = self.annotation
        max_depth = self.sampling_params['neighbor_max_depth']
        neighbors_dict = {0: [view]}

        for h in range(1, max_depth + 1):
            neighbors_dict[h] = []
            for prev_depth_view in neighbors_dict[h - 1]:
                already_counted = []
                for neighbors_list in neighbors_dict.values():
                    already_counted += neighbors_list
                for direction in self.sampling_params['neighbor_directions']:
                    next_view = ann[prev_depth_view][direction]
                    if next_view != '' and next_view not in already_counted:
                        neighbors_dict[h].append(next_view)

        del neighbors_dict[0]

        return neighbors_dict

    def sample_neighbors(self, neighbors):
        samples_per_depth = self.sampling_params['samples_per_depth']
        sampled_neighbors = []

        for depth, views in neighbors.items():
            num_samples_param = samples_per_depth[depth]
            num_views = len(views)
            num_samples = min(num_samples_param, num_views)
            all_indices = list(range(num_views))

            for idx in np.random.choice(all_indices, size=num_samples, replace=False):
                sampled_neighbors.append(views[idx])

        return sampled_neighbors

    def generate_triplets(self):
        pass

        """
            Generate train test clusters
            Sample multiple views per cluster in both train and test clusters
            Iterate over train test clusters
                Iterate over ref_view in sample_ref_views:
                    Generate Neighbors
                    Sample Neighbors
                    Iterate over sample_neighbors over generate_neighbors
                        Iterate over each bounding box
        """
        self.generate_train_test()
        self.sample_ref_views()
        proposals_per_img = self.sampling_params['proposals_per_img']

        triplet_train_test_folder = ['Train', 'Test']

        for set_idx, ref_views in enumerate([self.train_ref_views, self.test_ref_views]):

            set_folder = triplet_train_test_folder[set_idx]

            for ref_view in ref_views:
                self.logger.info(f'USING REFERENCE VIEW {ref_view}')

                tc1w, Rc1w = get_tR(ref_view, self.image_struct)
                if len(tc1w) == 0:
                    self.logger.debug(f'tR not available for img: {ref_view}')
                    print(f'tR not available for img: {ref_view}')
                    continue
                tc1w *= self.scale
                twc1, Rwc1 = camera_to_world_tR(tc1w, Rc1w)

                bboxes = self.load_pt_file(ref_view)[:proposals_per_img]

                if bboxes is None:
                    self.logger.exception(f'Proposal file: {ref_view} not found!!')
                    continue

                img_path = os.path.join(self.img_folder_path, ref_view)
                img = Image.open(img_path).resize(self.image_params['proposal_size'])
                depth_name = ref_view.split('.')[0][:-1] + '3.png'
                depth_path = os.path.join(self.depth_folder_path, depth_name)
                depth = Image.open(depth_path).resize(self.image_params['proposal_size'],
                                                      resample=Image.NEAREST)

                if self.plot_params['proposals'] or self.plot_params['save_proposals']:
                    fig, ax = bboxplot_in_img(img, bboxes, fontsize=7,
                                              return_fig=True,
                                              linewidth=2)
                    plt.title(f'Proposals for {ref_view}')
                    if self.plot_params['proposals']:
                        plt.show()

                    if self.plot_params['save_proposals']:

                        fig.savefig(os.path.join(self.plot_save_path,
                                                 f'proposals_{ref_view.split(".")[0]}.png'),
                                    bbox_inches='tight', dpi=300)
                    plt.close()
                x, y, z = generate_flat_xyz(depth)

                bboxes_px_idx = bbox_pixel_indices_list(np.array(bboxes),
                                                        x_flat=x,
                                                        y_flat=y,
                                                        z_flat=z,
                                                        filter_depth=False,
                                                        coordinates=False)

                pcl_cam1, _ = project_xyz_to_camera(x_flat=x, y_flat=y, z_flat=z,
                                                    center_x=self.cx_prop, center_y=self.cy_prop,
                                                    focal_x=self.fx_prop, focal_y=self.fy_prop)

                neighbor_views = self.generate_neighbors(view=ref_view)
                sampled_neighbors = self.sample_neighbors(neighbors=neighbor_views)

                triplet_dict = OrderedDict()

                for sampled_neighbor in sampled_neighbors:
                    triplet_dict[sampled_neighbor] = []

                    self.logger.info(f'\nCalculating for neighbor {sampled_neighbor}')
                    self.reproject_match_bboxes(ref_name=ref_view, ref_img=img, pcl_cam1=pcl_cam1,
                                                ref_bboxes=bboxes, ref_bbox_indices=bboxes_px_idx,
                                                neighbor_name=sampled_neighbor, twc1=twc1,
                                                Rwc1=Rwc1,
                                                output_list=triplet_dict[sampled_neighbor])

                pickle_filename = ref_view.split('.')[0] + '.pickle'
                pickle_path = os.path.join(self.triplet_save_path, set_folder, pickle_filename)

                with open(pickle_path, 'wb') as f:
                    pickle.dump(triplet_dict, f)

    def reproject_match_bboxes(self, ref_name, ref_img, pcl_cam1, ref_bboxes, ref_bbox_indices, neighbor_name, twc1,
                               Rwc1, output_list):
        """
            Reproject all Camera World pixels of reference image to Neighbor View 2d
            Load Neighbor Proposal bounding boxes
                Skip this neighbor if proposal not available
            For each bounding box in reference image
                If num pixels inside reprojected image < threshold
                    Skip this bounding box
                For each bounding box in neighbor image
                    Calculate iou
                Find max iou
                If max iou < threshold, skip
        """
        tc2w, Rc2w = get_tR(neighbor_name, self.image_struct)

        if len(tc2w) == 0:
            self.logger.debug(f'tR not available for neighbor {neighbor_name}')
            print(f'tR not available for neighbor {neighbor_name}')
            return None

        tc2w *= self.scale
        tc2c1, Rc2c1 = inter_camera_tR(twc1, Rwc1, tc2w, Rc2w)

        pcl_cam21 = np.matmul(Rc2c1, pcl_cam1) + tc2c1
        proj21 = project_camera_to_2d(pcl_cam21, center_x=self.cx_prop, center_y=self.cy_prop,
                                      focal_x=self.fx_prop, focal_y=self.fy_prop)

        neighbor_bboxes = self.load_pt_file(neighbor_name)

        if neighbor_bboxes is None:
            # print(f'File not found maybe because no proposals in image!!')
            self.logger.exception(
                f'Proposal file: {neighbor_name} not found!!')
            return None

        neighbor_bboxes = neighbor_bboxes[:self.sampling_params['proposals_per_neighbor_image']]

        neighbor_img_path = os.path.join(self.img_folder_path, neighbor_name)
        neighbor_img = Image.open(neighbor_img_path).resize(self.image_params['proposal_size'])

        for e, bbox_px_idx in enumerate(ref_bbox_indices):
            current_ref_bbox = ref_bboxes[e]

            proj_x = proj21[0][bbox_px_idx]
            proj_y = proj21[1][bbox_px_idx]
            proj_z = pcl_cam21[2][bbox_px_idx]

            inside_image = np.logical_and.reduce(
                (proj_x >= 0, proj_x < self.image_params['proposal_width'],
                 proj_y >= 0, proj_y < self.image_params['proposal_height'],
                 proj_z > 0))

            valid_projected_pxl = np.sum(inside_image)
            # FixMe: Include zero depths in count as well
            total_projected_pxl = len(bbox_px_idx)

            if valid_projected_pxl < self.match_params['min_valid_pixels']:
                # print(f'Skipping valid pixels = {valid_projected_pxl}')
                self.logger.debug(f'Skipping because valid pixels = {valid_projected_pxl}')
                continue

            proj_x = proj_x[inside_image]
            proj_y = proj_y[inside_image]

            matched_pixels_iou = []

            for neighbor_bbox_idx, neighbor_bbox in enumerate(neighbor_bboxes):
                area_bbox = (neighbor_bbox[2] - neighbor_bbox[0]) * (
                        neighbor_bbox[3] - neighbor_bbox[1])
                proj_x_inside_bbox = np.logical_and(
                    proj_x >= neighbor_bbox[0],
                    proj_x <= neighbor_bbox[2])
                proj_y_inside_bbox = np.logical_and(
                    proj_y >= neighbor_bbox[1],
                    proj_y <= neighbor_bbox[3])
                num_inside_pixels = np.sum(
                    np.logical_and(proj_x_inside_bbox, proj_y_inside_bbox))
                matched_pixels_iou.append(num_inside_pixels /
                                          (area_bbox + (total_projected_pxl - num_inside_pixels)))

            # TODO: Random sample for multiple same max_iou
            max_iou_idx = np.argmax(matched_pixels_iou)
            max_iou = matched_pixels_iou[max_iou_idx]
            self.logger.info(f'Max IOU = {max_iou}')

            pos_bbox = neighbor_bboxes[max_iou_idx]

            # PLOT REF AND POS BBOX
            # -------------------------------------------------------------------------
            if self.plot_params['ref_pos'] or self.plot_params['save_ref_pos'] or \
                    self.plot_params['triplet'] or self.plot_params['save_triplet']:

                fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(25, 14))
                ax1.imshow(ref_img)
                ax2.imshow(neighbor_img)

                _ = scatterplot_in_img(neighbor_img, [proj_x, proj_y],
                                       s=2, return_fig=True,
                                       fig=fig,
                                       ax=ax2)
                _ = bboxplot_in_img(neighbor_img,
                                    [pos_bbox],
                                    fig=fig, ax=ax2,
                                    numbering=False,
                                    return_fig=True, linewidth=3)
                _ = bboxplot_in_img(ref_img, [current_ref_bbox],
                                    fig=fig, ax=ax1,
                                    numbering=False,
                                    return_fig=True, linewidth=3
                                    )

                ax1.title.set_text(ref_name)
                ax2.title.set_text(neighbor_name)

                ref_name_raw = ref_name.split(".")[0]
                neighbor_name_raw = neighbor_name.split(".")[0]

                fig.suptitle(f'{ref_name_raw}_{neighbor_name_raw}_{e}\nIOU = {max_iou:.3f}')

                if self.plot_params['ref_pos']:
                    plt.show()

                if self.plot_params['save_ref_pos']:
                    fig.savefig(os.path.join(self.plot_save_path,
                                             f'{ref_name_raw}_{neighbor_name_raw}_{e}.png'),
                                dpi=300)
            # -------------------------------------------------------------------------

            if max_iou < self.match_params['min_pos_iou']:
                self.logger.info(f'No bbox matched in neighbor {neighbor_name} because small IOU')
                print('Not matched')
                continue

            neg_bbox = self.find_negative_bbox(bboxes=ref_bboxes, ref_bbox=current_ref_bbox)

            if neg_bbox is None:
                self.logger.debug(f'Could not find negative bbox!')
                print(f'Could not find negative bbox!')
            else:
                output_list.append([current_ref_bbox, pos_bbox, neg_bbox])

                # PLOT TRIPLET
                # -------------------------------------------------------------------------
                if self.plot_params['triplet'] or self.plot_params['save_triplet']:
                    print('Triplet')
                    _ = bboxplot_in_img(ref_img,
                                        [neg_bbox],
                                        fig=fig, ax=ax1,
                                        numbering=False, edgecolor='r',
                                        return_fig=True, linewidth=3)

                    fig.suptitle(f'Triplet_{ref_name_raw}_{neighbor_name_raw}_{e}\nIOU = {max_iou:.3f}')

                    if self.plot_params['triplet']:
                        print('plot triplet')
                        # FixMe: Plot using plt because fig.show doesnt wait to close window
                        fig.show()

                    if self.plot_params['save_triplet']:
                        fig.savefig(os.path.join(self.plot_save_path,
                                                 f'Triplet_{ref_name_raw}_{neighbor_name_raw}_{e}.png'),
                                    dpi=300)

                    plt.close()
                # -------------------------------------------------------------------------

        return None

    def find_negative_bbox(self, bboxes, ref_bbox):

        search_idx_list = list(range(len(bboxes)))
        # Shuffle the index list
        np.random.shuffle(search_idx_list)
        ref_bbox_area = (ref_bbox[2] - ref_bbox[0]) * (
                ref_bbox[3] - ref_bbox[1])

        found_neg = False

        for neg_bbox_idx in search_idx_list:
            neg_bbox = bboxes[neg_bbox_idx]

            intersect_xmin = max(ref_bbox[0], neg_bbox[0])
            intersect_ymin = max(ref_bbox[1], neg_bbox[1])
            intersect_xmax = min(ref_bbox[2], neg_bbox[2])
            intersect_ymax = min(ref_bbox[3], neg_bbox[3])
            intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
            neg_bbox_area = (neg_bbox[3] - neg_bbox[1]) * (neg_bbox[2] - neg_bbox[0])
            union_area = ref_bbox_area + neg_bbox_area - intersect_area
            iou = intersect_area / union_area

            if iou <= self.match_params['max_neg_iou']:
                # print(f'Neg bbox iou = {iou}')
                self.logger.debug(f'Neg bbox iou = {iou}')
                found_neg = True
                break

        if not found_neg:
            self.logger.debug(f'Did not find negative sample so skipping!!')
            return None

        return bboxes[neg_bbox_idx]


if __name__ == '__main__':
    DATASET_PATH = '/mnt/sda2/workspace/DATASETS/ActiveVision'
    PROPOSALS_PATH = '/home/sulabh/workspace-ubuntu/proposals/av_set1_train_coco'
    TRIPLET_PATH = '/home/sulabh/workspace-ubuntu/triplets_temp'
    scene = 'Home_003_1'
    CAMERA_INTRINSICS = {'fx': 1.0477637710998533e+03,
                         'fy': 1.0511749325842486e+03,
                         'cx': 9.5926120509632392e+02,
                         'cy': 5.2911546499433564e+02}

    SAMPLING_PARAMS = {
        'proposals_per_img': 100,
        'sample_per_cluster': 2,
        'neighbor_max_depth': 2,
        'neighbor_directions': ['rotate_ccw', 'forward', 'rotate_cw'],
        'samples_per_depth': {
            1: 1,
            2: 3
        },
        'proposals_per_neighbor_image': 100
    }

    IMAGE_PARAMS = {
        'org_width': 1920,
        'org_height': 1080,
        'proposal_size': (1333, 750),
        'proposal_width': 1333,
        'proposal_height': 750
        }
    IMAGE_PARAMS['x_scale_org_to_proposal'] = IMAGE_PARAMS['proposal_width']/IMAGE_PARAMS['org_width']
    IMAGE_PARAMS['y_scale_org_to_proposal'] = IMAGE_PARAMS['proposal_height']/IMAGE_PARAMS['org_height']
    MATCH_PARAMS = {
        'min_valid_pixels': 15,
        'min_pos_iou': 0.2,
        'max_neg_iou': 0.1
    }

    PLOT_PARAMS = {
        'clusters': False,
        'save_clusters': True,
        'proposals': False,
        'save_proposals': False,
        'ref_pos': False,
        'save_ref_pos': False,
        'triplet': False,
        'save_triplet': False
    }

    t = TripletGeneratorNode(dataset_path=DATASET_PATH, proposals_path=PROPOSALS_PATH,
                             triplet_save_path=TRIPLET_PATH, scene=scene,
                             camera_intrinsics=CAMERA_INTRINSICS, sampling_params=SAMPLING_PARAMS,
                             image_params=IMAGE_PARAMS, match_params=MATCH_PARAMS,
                             plot_params=PLOT_PARAMS)

    t.generate_triplets()
