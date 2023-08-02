'''
Author: aiwenjie aiwenjie20@outlook.com
Date: 2022-10-26 16:42:12
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-03 14:03:38
FilePath: /model_one/model_base/centroid_to_img.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import argparse
from tqdm import tqdm
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--root-features', type=str, default='/data3/data_haomo/m1/img_feature',
                    help='path to folder containing all the video folders with the features')
parser.add_argument('-i', '--root-imgs', type=str, default='/data3/data_haomo/m1/img/bd_1026',
                    help='path to folder containing all the video folders with the images corresponding to the features')
parser.add_argument('-c', '--centroid-file', type=str, default='/data3/data_haomo/m1/centroid_data/centroids-144.npy',
                    help='the .npy file containing all the centroids')
parser.add_argument('-s', '--save-file', type=str, default='/data/wangyue/model_one/config_service/centroid.json',
                    help='json file to save the centroid to image dictionary in')
args = parser.parse_args()

saved_features = args.root_features
saved_imgs = args.root_imgs
centroids = np.load(args.centroid_file)
save_path = args.save_file
centroid_map = {}

feature_paths = {}
feature_list = []

counter = 0


def img_path_from_centroid(features, centroids, img_dir):
    # vid_id = None
    for i in tqdm(range(counter)):
        min_dist = 2 ** 12 - 1
        features_id = None
        features_row = None
        for j in range(start_id, centroids.shape[0]):
            # 求范数
            centroid_dist = np.linalg.norm(features[i] - centroids[j], axis=1)
            centroid_min_dist = np.min(centroid_dist)
            if centroid_min_dist < min_dist:
                path = feature_paths[i]
                min_dist = centroid_min_dist
                # vid_id = path[path.index('/') + 1: path.rindex('/')]
                features_id = path[path.rindex('/') + 1: path.rindex('.')]
                # 显示最小值的下标
                features_row = j
        print(features_id, features_row)
        centroid_map[str(features_id)]=features_row
        # centroid_map[str(features_row)].append(os.path.join(img_dir,'img-{}-{:02}.jpg'.format(features_id, features_row)))

    

for root, dirs, files in tqdm(os.walk(saved_features)):
    for name in files:
        path = os.path.join(root, name)
        feature_list.append(np.load(path))
        feature_paths[counter] = path
        counter += 1


start_id = 0
try:
    # centroid_map = json.load(open(save_path, 'r'))
    # start_id = len(centroid_map)
    centroid_map={}
    print('starting at centroid', start_id)
except:
    print('starting with empty centroid_map')

img_path_from_centroid(feature_list, centroids, saved_imgs)
print('saving centroids and corresponding images')

json.dump(centroid_map, open(save_path, 'w'), sort_keys=True, indent=4)

# 聚类的评价指标 CH指数