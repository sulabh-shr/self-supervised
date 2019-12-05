import os
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import pickle

class BboxEvaluator:
    SCENES = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1',
              'Home_003_2', 'Home_004_1', 'Home_004_2', 'Home_005_1',
              'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
              'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1',
              'Home_014_2', 'Home_015_1', 'Home_016_1', 'Office_001_1']

    def __init__(self, dataset_root, proposals_path, image_params, evaluation_params):
        self.proposals_path = proposals_path
        self.dataset_root = dataset_root
        self.evaluation_params = evaluation_params
        self.image_params = image_params

        self.scale_x_proposal_to_org = image_params['org_width'] / image_params['proposal_width']
        self.scale_y_proposal_to_org = image_params['org_height'] / image_params['proposal_height']

        self.ann = {}
        self.img_folder_map = {}

        self.create_folder_map()

        self.matched = None
        self.total = None

    def create_folder_map(self):

        for scene in self.SCENES:
            scene_path = os.path.join(self.dataset_root, scene)
            scene_img_path = os.path.join(scene_path, 'jpg_rgb')
            images = os.listdir(scene_img_path)

            for img in images:
                self.img_folder_map[img] = scene

            ann_path = os.path.join(scene_path, 'annotations.json')

            with open(ann_path, 'r') as f:
                self.ann.update(json.load(f))

    def load_pt_file(self, filename):
        if not filename.endswith('.pt'):
            filename = filename.split('.')[0] + '.pt'

        try:
            bboxes = torch.load(os.path.join(self.proposals_path, filename))[0]

        # Get the bounding boxes
        except FileNotFoundError:
            print(f'File {filename} not found!!')
            return None

        bboxes_coords = bboxes.bbox.cpu().numpy()

        return bboxes_coords

    def evaluate(self):

        matched = defaultdict(lambda: defaultdict(int))
        total_gt = defaultdict(lambda: defaultdict(int))

        for idx, proposal_file in tqdm(enumerate(os.listdir(self.proposals_path)),
                                       total=len(os.listdir(self.proposals_path))):
            img_file = proposal_file.split('.')[0] + '.jpg'
            bboxes = self.load_pt_file(proposal_file)[:self.evaluation_params['proposals_per_img']]

            # SCALE TO ORIGINAL INPUT IMAGE SIZE
            bboxes[:, [0, 2]] *= self.scale_x_proposal_to_org
            bboxes[:, [1, 3]] *= self.scale_y_proposal_to_org

            gt_bboxes = self.ann[img_file]['bounding_boxes']

            if len(gt_bboxes) == 0:
                continue

            scene = self.img_folder_map[img_file]

            for gt_bbox in gt_bboxes:

                label = gt_bbox[4]

                total_gt[scene][label] += 1
                gt_bbox_area = (gt_bbox[3] - gt_bbox[1]) * (gt_bbox[2] - gt_bbox[0])

                for bbox in bboxes:

                    intersect_xmin = max(bbox[0], gt_bbox[0])
                    intersect_ymin = max(bbox[1], gt_bbox[1])
                    intersect_xmax = min(bbox[2], gt_bbox[2])
                    intersect_ymax = min(bbox[3], gt_bbox[3])
                    intersect_area = max(0, (intersect_xmax - intersect_xmin) * (
                                intersect_ymax - intersect_ymin))

                    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                    union_area = gt_bbox_area + bbox_area - intersect_area
                    iou = intersect_area / union_area

                    if iou >= self.evaluation_params['iou_thresh']:
                        matched[scene][label] += 1

                        break

                # Make sure there is 0 value if No match at all
                matched[scene][label] += 0

        self.matched = matched
        self.total = total_gt

        return matched, total_gt

    def report(self):
        scenes = sorted(list(self.matched.keys()))

        print(f'\nProposals path : {self.proposals_path}')
        print(f'Using params   : \n{self.evaluation_params}\n')
        print('-'*70)

        for scene in scenes:
            print(f'\nScene = {scene}')
            available_labels = sorted(list(self.total[scene].keys()))

            scene_match_bboxes = 0
            scene_gt_bboxes = 0

            for label in available_labels:
                label_matched = self.matched[scene][label]
                label_total = self.total[scene][label]

                scene_match_bboxes += label_matched
                scene_gt_bboxes += label_total

                print(f'For LABEL = {label:2d} | Recall = {label_matched/label_total:.3f}| '
                      f'Total = {label_total:<4d} | Matched = {label_matched:<4d}')

            print(f'Total Recall = {scene_match_bboxes/scene_gt_bboxes:.3f}')

        print('-'*70)


if __name__ == '__main__':
    DATASET_PATH = '/mnt/sda2/workspace/DATASETS/ActiveVision'
    PROPOSALS_PATH = '/home/sulabh/workspace-ubuntu/proposals/av_set1_train_coco'

    IMAGE_PARAMS = {
        'org_width': 1920,
        'org_height': 1080,
        'proposal_size': (1333, 750),
        'proposal_width': 1333,
        'proposal_height': 750
    }

    EVALUATION_PARAMS = {
        'proposals_per_img': 50,
        'iou_thresh': 0.7
    }
    BE = BboxEvaluator(dataset_root=DATASET_PATH, proposals_path=PROPOSALS_PATH,
                       image_params=IMAGE_PARAMS, evaluation_params=EVALUATION_PARAMS)

    matched, total = BE.evaluate()

    BE.report()

