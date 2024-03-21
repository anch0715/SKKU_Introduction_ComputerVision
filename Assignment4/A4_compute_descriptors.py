import glob
import os
import struct
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def clustering(local_des):
    # 4-branches, 4-depth
    brief = deepcopy(local_des)
    des = np.array(sum(local_des, []))
    kmeans = KMeans(n_clusters=4, random_state=0).fit(des)
    first_c = kmeans.labels_
    k = 0
    #depth-1 clustering
    for i in range(len(local_des)):
      for j in range(len(local_des[i])):
        brief[i][j] = first_c[k]
        k+=1

    for depth in range(2, 5):
        for b in range(4**(depth-1)):
            #각 라벨에 대해 다시 clustering 진행
            des = [local_des[i][j] for i in range(len(local_des)) for j in range(len(local_des[i])) if brief[i][j]==b]
            rec = [(i,j) for i in range(len(local_des)) for j in range(len(local_des[i])) if brief[i][j]==b]
            kmeans = KMeans(n_clusters=4, random_state=2).fit(des)
            n_c = kmeans.labels_
            k=0
            for (i,j) in rec:
                brief[i][j] = n_c[k]+4*b
                k+=1

    return brief

def BOW(brief):
    his = np.zeros((len(brief), 256))
    for i in range(len(brief)):
        a = np.array(brief[i])
        count = np.unique(a, return_counts=True)
        for k, v in zip(count[0], count[1]):
            his[i, k] = v

    return his


# path

bin = []
file_path = './sift'

sorted_lst = sorted(glob.glob(os.path.join(file_path, '*')))
for fname in sorted_lst:
    with open(fname, 'rb') as f:
        bin.append(f.read())

sift = [b for b in bin]

des = []
for i in sift:
    brief = [b for b in i]
    des.append(brief)

sift_des = [list(np.array(d).reshape((-1, 128))) for d in des]

brief = clustering(sift_des)
global_des = BOW(brief)


global_des = np.float32(global_des)
N = len(sift_des)
D = 256


f = open("A4_2019314962.des", 'wb')
f.write(struct.pack('i', N))
f.write(struct.pack('i', D))
for i in range(len(global_des)):
    for j in range(256):
        f.write(struct.pack('f', global_des[i,j]))
f.close()
