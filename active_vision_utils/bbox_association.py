import numpy as np

from matlab_utils import get_tR
from coordinate_utils import generate_flat_xyz, bbox_pixel_indices_list
from projection import project_xyz_to_camera, camera_to_world_tR, \
    inter_camera_tR, project_camera_to_2d


class BboxAssoc:

    def __init__(self, img_name, depth1, width2, height2, image_struct, scale,
                 img_names, center_x, center_y, focal_x, focal_y):

        self.img_name = img_name

        self.depth1 = depth1
        self.width2 = width2
        self.height2 = height2
        self.image_struct = image_struct
        self.scale = scale
        self.img_names = img_names
        self.center_x = center_x
        self.center_y = center_y
        self.focal_x = focal_x
        self.focal_y = focal_y

        self.x, self.y, self.z = generate_flat_xyz(depth1)

        self.t = None
        self.R = None
        self.t_list = []
        self.R_list = []
        self.pcl_cam = None
        self.pcl_cam2_list = None
        self.xyz_list = []
        self.bbox_idx_list = []
        # self.load_parameters()

    def load_parameters(self):
        self.t, self.R = get_tR(self.img_name, self.image_struct)

        for img_name in self.img_names:
            t, R = get_tR(img_name, self.image_struct)
            self.t_list.append(t)
            self.R_list.append(R)

        self.t = self.t * self.scale
        self.t_list = np.array(self.t_list) * self.scale
        self.R_list = np.array(self.R_list)

    def unproject_img(self):
        self.pcl_cam, _ = project_xyz_to_camera(x_flat=self.x, y_flat=self.y,
                                                z_flat=self.z,
                                                center_x=self.center_x,
                                                center_y=self.center_y,
                                                focal_x=self.focal_x,
                                                focal_y=self.focal_y,
                                                filter_depth=False)

    def cam_to_world(self):
        self.t, self.R = camera_to_world_tR(self.t, self.R)

    def inter_camera_tR(self):
        self.t_list, self.R_list = inter_camera_tR(twc1=self.t, Rwc1=self.R,
                                                   tc2w=self.t_list,
                                                   Rc2w=self.R_list)

    def cam1_to_cam2(self):
        # (Num second images, 3, num_pts)
        self.pcl_cam2_list = np.matmul(self.R_list, self.pcl_cam) + self.t_list

    def unproject_pcl_list(self):
        z_list = self.pcl_cam2_list[:, 2, :]
        x_list = self.pcl_cam2_list[:, 0, :] / z_list + self.center_x
        y_list = self.pcl_cam2_list[:, 1, :] / z_list + self.center_y

        for x, y, z in zip(x_list, y_list, z_list):
            self.xyz_list.append(np.array((x, y, z)))

    def get_bbox_idx(self, bboxes):
        return bbox_pixel_indices_list(np.array(bboxes), x_flat=self.x,
                                       y_flat=self.y, z_flat=self.z,
                                       filter_depth=True,
                                       coordinates=False)

    def reprojected_bbox_px(self, bboxes):
        # FIXME: empty array
        bbox_idx_list = self.get_bbox_idx(bboxes)
        bbox_px_per_img = []

        for x, y, z in self.xyz_list:
            bbox_img = []
            for bbox_idx in bbox_idx_list:
                bbox_img.append(
                    np.array((x[bbox_idx], y[bbox_idx], z[bbox_idx])))
            bbox_px_per_img.append(bbox_img)
        return bbox_px_per_img

    def reproject(self, bboxes):
        self.load_parameters()
        self.unproject_img()
        self.cam_to_world()
        self.inter_camera_tR()
        self.cam1_to_cam2()
        self.unproject_pcl_list()
        return self.reprojected_bbox_px(bboxes)


if __name__ == '__main__':
    from defaults import *
    from matlab_utils import load_image_struct
    from plot_utils import plot_in_image
    from PIL import Image
    import os

    input_img_name = '000110000010101.jpg'
    input_depth_name = input_img_name.split('.')[0][:-1] + '3.png'
    input_img = Image.open(os.path.join(root_path, 'jpg_rgb', input_img_name))
    input_depth = Image.open(
        os.path.join(root_path, 'high_res_depth', input_depth_name))

    img_list = ['000110000120101.jpg', '000110000860101.jpg']
    image_struct, scale = load_image_struct(root_path)

    bbox_assoc = BboxAssoc(img_name=input_img_name, depth1=input_depth,
                           width2=None, height2=None, image_struct=image_struct,
                           scale=scale, img_names=img_list, center_x=cx,
                           center_y=cy, focal_x=fx, focal_y=fy)
    bboxes = np.array([[935, 505, 1000, 635], [755, 550, 815, 650]])
    bboxes_coords = [bboxes[:, np.arange(0, len(bboxes[0]), 2)].reshape(-1),
                     bboxes[:, np.arange(1, len(bboxes[0]), 2)].reshape(-1)]
    print(bboxes_coords)
    plot_in_image(input_img, coordinates=bboxes_coords, mode='2n')
    bbox_per_img_per_box = bbox_assoc.reproject(bboxes)
    # print(reproj)

    img2 = Image.open(os.path.join(root_path, 'jpg_rgb', img_list[0]))
    img3 = Image.open(os.path.join(root_path, 'jpg_rgb', img_list[1]))

    for bbox_img, i in zip(bbox_per_img_per_box, [img2, img3]):
        for bbox in bbox_img:
            print(bbox.shape)
            out_cooords = project_camera_to_2d(bbox)
            plot_in_image(i, out_cooords, mode='2n')
