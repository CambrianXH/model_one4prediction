'''
Author: aiwenjie aiwenjie20@outlook.com
Date: 2022-10-24 16:53:08
LastEditors: aiwenjie aiwenjie20@outlook.com
LastEditTime: 2022-11-07 16:07:24
FilePath: /model_one/model_base/minibatch_hkmeans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from hkmeans_minibatch.hkmeans  import hkmeans
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root-feature_path', type=str,default='/data3/data_haomo/m1/img_feature',
                        help='path to folder containing all the video folders with the features')
    parser.add_argument('-p', '--features-prefix', type=str, default='H',
                        help='prefix that contains the desired files to read')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='batch_size to use for the minibatch kmeans')
    parser.add_argument('-s', '--save-dir', type=str, default='/data3/data_haomo/m1/cluster_data1',
                        help='save directory for hierarchical kmeans vectors')
    parser.add_argument('-c', '--centroid-dir', type=str,default='/data3/data_haomo/m1/centroid_data',
                        help='directory to save the centroids in')
    parser.add_argument('-hr', '--hierarchies', type=int, required=True,
                    help='number of hierarchies to run the kmeans on')
    parser.add_argument('-k', '--clusters', type=int, required=True,
                    help='number of clusters for each part of the hierarchy')

    parser.add_argument('-e', '--epochs', type=int, default=15,
                    help='number of epochs to run the kmeans for each hierarchy')


    args = parser.parse_args()

    root = args.root_feature_path
    prefix = args.features_prefix
    batch_size = args.batch_size
    save_dir = args.save_dir
    centroid_dir = args.centroid_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(centroid_dir):
        os.mkdir(centroid_dir)
    h = args.hierarchies
    k = args.clusters
    epochs = args.epochs
    hkmeans(root, prefix, h, k, batch_size, epochs, save_dir, 'vecs', centroid_dir)


if __name__ == "__main__":
    main()