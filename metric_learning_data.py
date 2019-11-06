import os
import torch
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


class ActiveVisionTriplet(Dataset):

    def __init__(self, dataset_root, triplet_root, instance, size):
        self.dataset_root = dataset_root
        self.triplet_root = triplet_root
        if instance == 'instance1':
            with open(os.path.join(dataset_root, 'coco_annotations',
                                   'instances_set_1_train.json')) as f:
                self.img_folder_map = json.load(f)['img_folder_map']
        else:
            raise ValueError(f'Invalid instance: {instance}')
        self.size = size

        self.pickle_names = [i for i in sorted(os.listdir(triplet_root)) if
                             i.endswith('pickle')]
        self.pickles_dict = {}
        self.triplets = []

        for pickle_name in self.pickle_names:
            img_name = pickle_name.split('.')[0] + '.jpg'

            with open(os.path.join(triplet_root, pickle_name), 'rb') as f:
                content = pickle.load(f)
                self.pickles_dict[img_name] = content

            for neighbor_img_name, neighbor_triplets in content.items():
                for neighbor_triplet in neighbor_triplets:
                    triplet_dict = {
                        'ref': [img_name, neighbor_triplet[0]],
                        'pos': [neighbor_img_name, neighbor_triplet[1]],
                        'neg': [img_name, neighbor_triplet[2]]
                    }
                    self.triplets.append(triplet_dict)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]

        ref_img_folder = self.img_folder_map[triplet['ref'][0]]
        pos_img_folder = self.img_folder_map[triplet['pos'][0]]

        ref_bbox = triplet['ref'][1]
        pos_bbox = triplet['pos'][1]
        neg_bbox = triplet['neg'][1]
        print(ref_bbox)

        # ref_bbox = [ref_bbox[0], ref_bbox[2]-ref_bbox[0], ref_bbox[3], ref_bbox[3]-ref_bbox[1]]

        ref_img = Image.open(os.path.join(self.dataset_root, ref_img_folder,
                                          'jpg_rgb',
                                          triplet['ref'][0])).resize(self.size)
        pos_img = Image.open(os.path.join(self.dataset_root, pos_img_folder,
                                          'jpg_rgb',
                                          triplet['pos'][0])).resize(self.size)
        return ref_img.crop(ref_bbox), pos_img.crop(pos_bbox), ref_img.crop(
            neg_bbox)

    def visualize_triplet(self, idx):
        import matplotlib.pyplot as plt
        from matplotlib import patches
        triplet = self.triplets[idx]

        for k, v in triplet.items():
            print(k, v)

        ref_img_folder = self.img_folder_map[triplet['ref'][0]]
        pos_img_folder = self.img_folder_map[triplet['pos'][0]]

        ref_bbox = triplet['ref'][1]
        pos_bbox = triplet['pos'][1]
        neg_bbox = triplet['neg'][1]

        # TODO: Make negative image independent of ref image
        ref_img = Image.open(os.path.join(self.dataset_root, ref_img_folder,
                                          'jpg_rgb',
                                          triplet['ref'][0])).resize(self.size)
        pos_img = Image.open(os.path.join(self.dataset_root, pos_img_folder,
                                          'jpg_rgb',
                                          triplet['pos'][0])).resize(self.size)

        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(25, 14))

        ax1.imshow(ref_img)
        ax2.imshow(pos_img)

        ref_rect = patches.Rectangle((ref_bbox[0], ref_bbox[1]),
                                     ref_bbox[2] - ref_bbox[0],
                                     ref_bbox[3] - ref_bbox[1],
                                     linewidth=1, edgecolor='b',
                                     facecolor='none')
        ax1.add_patch(ref_rect)
        ax1.text(ref_bbox[0], ref_bbox[1], 'ref', fontsize=10)

        pos_rect = patches.Rectangle((pos_bbox[0], pos_bbox[1]),
                                     pos_bbox[2] - pos_bbox[0],
                                     pos_bbox[3] - pos_bbox[1],
                                     linewidth=1, edgecolor='b',
                                     facecolor='none')
        ax2.add_patch(pos_rect)
        ax2.text(pos_bbox[0], pos_bbox[1], 'pos', fontsize=10)

        neg_rect = patches.Rectangle((neg_bbox[0], neg_bbox[1]),
                                     neg_bbox[2] - neg_bbox[0],
                                     neg_bbox[3] - neg_bbox[1],
                                     linewidth=1, edgecolor='r',
                                     facecolor='none')
        ax1.add_patch(neg_rect)
        ax1.text(neg_bbox[0], neg_bbox[1], 'neg', fontsize=10)
        plt.show()

    def visualize_multi_triplets(self, num=None, random=True):
        import numpy as np

        num_triplets = len(self.triplets)

        if num is None:
            num = num_triplets

        all_indices = list(range(num_triplets))

        if random:
            all_indices = np.random.choice(all_indices, num_triplets,
                                           replace=False)

        for idx in all_indices[:num]:
            print(f'Index = {idx}')
            self.visualize_triplet(idx)


if __name__ == '__main__':

    a = ActiveVisionTriplet('/mnt/sda2/workspace/DATASETS/ActiveVision',
                            '/home/sulabh/workspace-ubuntu/triplets',
                            instance='instance1', size=(1333, 750))

    # a.visualize_multi_triplets(num=3, random=True)
    a.visualize_triplet(100)

    ref, pos, neg = a.__getitem__(100)
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    ax1.imshow(ref)
    ax2.imshow(pos)
    ax3.imshow(neg)
    plt.show()
