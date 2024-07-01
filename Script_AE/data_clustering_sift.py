import numpy as np
import argparse
import struct
from sklearn.cluster import KMeans
import sklearn.metrics


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.i8bin)")
    parser.add_argument("--dst", help="The output file prefix (.i8bin)")
    return parser.parse_args()


if __name__ == "__main__":
    clusters = 2
    args = process_args()

    # Read topk vector one by one
    vecs = []
    with open(args.src, "rb") as f:
        row = int.from_bytes(f.read(4), byteorder='little')
        dim = int.from_bytes(f.read(4), byteorder='little')

        for _ in range(row):
            vec = np.frombuffer(f.read(dim), dtype=np.int8)
            vecs.append(vec)

    # Clustering vectors
    vecs = np.array(vecs)
    assert vecs.shape[0] == row
    print("vecs.shape:", vecs.shape)
    estimator = KMeans(n_clusters=clusters)
    estimator.fit(vecs)
    label_pred = estimator.labels_

    print("Clustering finished")
    # print(sklearn.metrics.silhouette_score(vecs, label_pred, metric='euclidean'))

    # Generate result
    vec_list = [[] for _ in range(clusters)]
    vec_num_list = [0] * clusters

    with open(args.src, "rb") as f:
        row = int.from_bytes(f.read(4), byteorder='little')
        dim = int.from_bytes(f.read(4), byteorder='little')

        for j in range(row):
            vec = np.frombuffer(f.read(dim), dtype=np.int8)
            cluster_idx = label_pred[j]
            vec_list[cluster_idx].append(vec.tobytes())
            vec_num_list[cluster_idx] += 1
            if (j + 1) % 100000 == 0:
                print(j + 1)

    print("Cluster_result: ", vec_num_list)

    for i in range(clusters):
        with open(args.dst + str(i), "wb") as f:
            f.write(struct.pack('i', vec_num_list[i]))
            f.write(struct.pack('i', dim))
            for vec in vec_list[i]:
                f.write(vec)

    with open(args.dst + str(clusters), "wb") as f:
        f.write(struct.pack('i', row))
        f.write(struct.pack('i', dim))
        for i in range(clusters):
            for vec in vec_list[i]:
                f.write(vec)
