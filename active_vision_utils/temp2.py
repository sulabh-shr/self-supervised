import os
from PIL import Image
import matplotlib.pyplot as plt
from active_vision_utils.matlab_utils import get_tR, load_image_struct

data_root = '/mnt/sda2/workspace/DATASETS/ActiveVision/'
img_folder = 'Home_003_2'
img_name = '000320000030101.jpg'
img_str_path = os.path.join(data_root, img_folder, 'jpg_rgb', img_name)
img = Image.open(img_str_path)
# img.show()
plt.imshow(img)
plt.show()

img2 = img.crop((0, 450, 230, 900))
plt.imshow(img2)
plt.show()

plt.imshow(img)
plt.show()
#
#
# ist, s = load_image_struct(img_str_path)
# # print(i)
# print(len(ist))
# count = 0
#
# for i in ist:
#     row = i
#     if len(row[1]) == 0:
#         print(row[0][0], row[1])
#         count+=1
# print(count)
#     # if row[1] == []:
#     #     count += 1
#     # print(row[1], row[2])
# # t, R = get_tR('000320000020101.jpg', ist)
# # print(t, R)