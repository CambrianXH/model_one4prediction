import matplotlib.pyplot as plt
import argparse


from pydoc import pathdirs
import numpy as np
from numpy import *
from tqdm import tqdm
import os
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn import metrics



def minibatch_kmeans(root, prefix, k, batch_size, epochs):
    """
    docstring
    """
    paths = []
    for root, dirs, files in tqdm(os.walk(root)):
        print(len(files))
        for name in files:
            if name.find(prefix) != -1:
                paths.append(os.path.join(root, name))
        print(len(paths))
    kmeans = MiniBatchKMeans(n_clusters=k,batch_size=batch_size)
    vectors = None
    
    score = []
    print("starting kmeans")
    for i in range(epochs):
        ch_score = []
        print("epoch:", i)
        for path in tqdm(paths):
            if vectors is None:
                vectors = np.load(path).astype(float)
            else:
                vectors = np.concatenate([vectors, np.load(path)])
            if vectors.shape[0] >= batch_size:
                vectors = vectors[:batch_size, :]
                kmeans.partial_fit(vectors)  
                if len(np.unique(kmeans.labels_)) !=1:
                    ch_score.append(metrics.calinski_harabasz_score(vectors,kmeans.labels_))
                vectors = None
        if vectors is not None and vectors.shape[0] >= k:
            kmeans.partial_fit(vectors)
            vectors = None
        if vectors is not None and vectors.shape[0] < k:
            vectors=np.vstack((vectors,np.zeros((k-vectors.shape[0],vectors.shape[1])))) 
            kmeans.fit(vectors)
            vectors = None
        if vectors is  None:
            vectors=np.zeros((k,1792))
            kmeans.fit(vectors)
            vectors = None

        score.append(mean(ch_score))


    print("labelling data")
    labelled_data = {}
    for path in tqdm(paths):
        labelled_data[path] = list(kmeans.predict(np.load(path).astype(float)))
    return kmeans.cluster_centers_, labelled_data,mean(score)


def save_sorted_vectors(centroids, labelled_data, batch_size, save_dir, save_prefix):
    k = centroids.shape[0]
    save_path = os.path.join(save_dir, save_prefix) + '-{}-Id:{}'
    for i in range(k):
        sorted_vecs = []
        counter = 1
        for key in tqdm(labelled_data):
            pred_centroids = labelled_data[key]
            vectors = np.load(key)
            for j in range(len(pred_centroids)):
                if (pred_centroids[j] == i).all() and j< vectors.shape[0]:
                    sorted_vecs.append(np.expand_dims(vectors[j], axis=0))
                    if len(sorted_vecs) == batch_size:
                        sorted_vecs = np.concatenate(sorted_vecs)
                        np.save(save_path.format(i, counter), sorted_vecs)
                        sorted_vecs = []
                        counter += 1

        if sorted_vecs != []:
            sorted_vecs = np.concatenate(sorted_vecs)
            np.save(save_path.format(i, counter), sorted_vecs)
            sorted_vecs = []


def delete_used_files(root, prefix):
    print("deleting finished files")
    for root, dirs, files in tqdm(os.walk(root)):
        for name in files:
            if name.find(prefix) != -1:
                os.remove(os.path.join(root, name))

def hkmeans(root, prefix, h, k, batch_size, epochs, save_dir, save_prefix, centroid_dir):
    
    counter = 1
    def hkmeans_recursive(root, prefix, h, k, batch_size, epochs, save_dir, save_prefix, centroid_dir, cur_h=1):
        nonlocal counter
        print("Current H:", cur_h)
        print(prefix)
        if cur_h != h:
            centroids, labelled_data,ch_score = minibatch_kmeans(root, prefix, k, batch_size, epochs)
            print("minibatch kmeans done!")
            save_sorted_vectors(centroids, labelled_data, batch_size, save_dir, save_prefix)
            save_prefix += '-{}'
            for i in range(k):
                hkmeans_recursive(save_dir, save_prefix.format(i) + '-', h, k, batch_size, epochs, save_dir,
                                save_prefix.format(i), centroid_dir, cur_h=cur_h + 1)
                delete_used_files(save_dir, save_prefix.format(i) + '-')
        else:
            centroids, labelled_data, ch_score = minibatch_kmeans(root, prefix, k, batch_size, epochs)
            print("minibatch kmeans done!")
            # np.save(os.path.join(centroid_dir, 'centroids-{}'.format(counter)), centroids)
            counter += 1
        return ch_score
        

    return hkmeans_recursive(root, prefix, h, k, batch_size, epochs, save_dir, save_prefix, centroid_dir)

def count_path(root, prefix):
    paths = []
    for root, dirs, files in tqdm(os.walk(root)):
        print(len(files))
        for name in files:
            if name.find(prefix) != -1:
                paths.append(os.path.join(root, name))
    return len(paths)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root-feature_path', type=str,default='/data3/data_haomo/m1/clusters/1115/img_feature',
                        help='path to folder containing all the video folders with the features')
    parser.add_argument('-p', '--features-prefix', type=str, default='H',
                        help='prefix that contains the desired files to read')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='batch_size to use for the minibatch kmeans')
    parser.add_argument('-s', '--save-dir', type=str, default='/data3/data_haomo/m1/clusters/1115/cluster_data',
                        help='save directory for hierarchical kmeans vectors')
    parser.add_argument('-c', '--centroid-dir', type=str,default='/data3/data_haomo/m1/clusters/1115/centroid_data',
                        help='directory to save the centroids in')
    parser.add_argument('-hr', '--hierarchies', type=int, default=3,
                    help='number of hierarchies to run the kmeans on')
    parser.add_argument('-k', '--clusters', type=int, default=15,
                    help='number of clusters for each part of the hierarchy')

    parser.add_argument('-e', '--epochs', type=int, default=1,
                    help='number of epochs to run the kmeans for each hierarchy')


    args = parser.parse_args()

    root = args.root_feature_path
    prefix = args.features_prefix
    batch_size = args.batch_size
    save_dir = args.save_dir
    centroid_dir = args.centroid_dir
    h = args.hierarchies
    k = args.clusters
    epochs = args.epochs

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.centroid_dir):
        os.mkdir(args.centroid_dir)


    score_=[]
    for k1 in range(5,20):
        score_.append(hkmeans(root, prefix, h, k1, batch_size, epochs, save_dir, 'vecs', centroid_dir))
        print("................................................")
    fig,ax = plt.subplots()
    ax.plot(np.arange(5,20),score_)
    
    plt.savefig("/data/wangyue/model_one/experiment/mmrt_v1.1/visz/ch_score/ch_score.jpg")



if __name__ == "__main__":
    main()



