import os
from PIL import Image

from active_vision_utils.parameters import *
from active_vision_utils.projection import *
from active_vision_utils.coordinate_utils import *
from active_vision_utils.plot_utils import *
from active_vision_utils.matlab_utils import *

# img1_name = '000110000860101.jpg'
# depth1_name = img1_name.split('.')[0][:-1] + '3.png'
# img2_name = '000110000010101.jpg'
# depth2_name = img2_name.split('.')[0][:-1] + '3.png'

dataset_root = '/mnt/sda2/workspace/DATASETS/ActiveVision'
img_folder = 'Home_006_1'

img1_name = '000610001320101.jpg'
depth1_name = img1_name.split('.')[0][:-1] + '3.png'
neighbors = ['000610001310101.jpg', '000610000240101.jpg', '000610001210101.jpg', '000610002400101.jpg']
img2_name = neighbors[0]
img3_name = neighbors[1]
depth2_name = img2_name.split('.')[0][:-1] + '3.png'

img1 = Image.open(os.path.join(dataset_root, img_folder, 'jpg_rgb', img1_name)).resize((1333, 750))
depth1 = Image.open(os.path.join(dataset_root, img_folder,  'high_res_depth', depth1_name)).resize((1333, 750))
img2 = Image.open(os.path.join(dataset_root, img_folder, 'jpg_rgb', img2_name)).resize((1333, 750))
depth2 = Image.open(os.path.join(dataset_root, img_folder, 'high_res_depth', depth2_name)).resize((1333, 750))
d = np.array(depth1)
print(d[d!=0].reshape(-1))

image_struct, scale = load_image_struct(os.path.join(dataset_root, img_folder))

tc1w, Rc1w = get_tR(img1_name, image_struct)
tc2w, Rc2w = get_tR(img2_name, image_struct)
tc3w, Rc3w = get_tR(img3_name, image_struct)
tc1w = tc1w * scale
tc2w = tc2w * scale
tc3w = tc3w * scale

t_list = np.array([tc2w, tc3w])
R_list = np.array([Rc2w, Rc3w])

twc1, Rwc1 = camera_to_world_tR(tc1w, Rc1w)

tc2c1, Rc2c1 = inter_camera_tR(twc1, Rwc1, tc2w, Rc2w)
tc3c1, Rc3c1 = inter_camera_tR(twc1, Rwc1, tc3w, Rc3w)
t2_list, R2_list = inter_camera_tR(twc1, Rwc1, t_list, R_list)


# --------------------------------------------------------------

# plot_in_image(img1, [[1030, 630], [1135, 820], [1453, 522], [1590, 820],
#                     [595, 550], [632, 659]], mode='n2', numbering=False)

# bboxes = [[1030, 630, 1135, 820], [1453, 522, 1590, 820],
#           [595, 550, 632, 659]]
bboxes = np.array([[225, 55, 340, 175], [935, 500, 1000, 635], [755, 550, 815, 650], [180, 530, 225, 635],
          [1035, 645, 1100, 690]], dtype=float)
bboxes[1:, [0, 2]] *= 1333/1920
bboxes[1:, [1, 3]] *= 750/1080

bboxplot_in_img(img1, bboxes)

# GENERATE XYZ
x, y, z = generate_flat_xyz(depth1)

# GET INDICES OF EACH BOUNDING BOX'S PIXELS
bbox_px_idx = bbox_pixel_indices_list(np.array(bboxes), x_flat=x, y_flat=y,
                                      z_flat=z, filter_depth=True,
                                      coordinates=False)
for i in bbox_px_idx:
    print(min(z[i]), max(z[i]), sum(z[i])/len(z[i]))
    scatterplot_in_img(img1, coordinates=(x[i], y[i]), s=10)

x_scale = 1333/1920
y_scale = 750/1080
cx *= x_scale
cy *= y_scale
fx *= x_scale
fy *= y_scale

pcl_cam1, _ = project_xyz_to_camera(x_flat=x, y_flat=y, z_flat=z, center_x=cx,
                                    center_y=cy, focal_x=fx, focal_y=fy)
pcl_cam21 = np.matmul(Rc2c1, pcl_cam1) + tc2c1
pcl_cam21_list = np.matmul(R_list, pcl_cam1) + t_list
proj21 = project_camera_to_2d(pcl_cam21, center_x=cx, center_y=cy, focal_x=fx,
                              focal_y=fy)


for i in bbox_px_idx:
    scatterplot_in_img(img2, coordinates=(proj21[0][i], proj21[1][i]),
                       s=2)

