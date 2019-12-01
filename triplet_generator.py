import os
import abc
import torch


class TripletGenerator(abc.ABC):

    def __init__(self, dataset_path, proposals_path, triplet_save_path):
        self.dataset_path = dataset_path
        self.proposals_path = proposals_path
        self.triplet_save_path = triplet_save_path

    @abc.abstractmethod
    def sample_ref_views(self):
        pass

    @abc.abstractmethod
    def generate_neighbors(self):
        pass

    @abc.abstractmethod
    def sample_neighbors(self):
        pass

    @abc.abstractmethod
    def generate_triplets(self):
        pass

    @abc.abstractmethod
    def generate_train_test(self):
        pass

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
