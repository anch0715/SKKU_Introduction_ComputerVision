import numpy as np
import cv2
import time
import random
import compute_avg_reproj_error as error


def compute_F_raw(M):

    img1_coord = M[:, :2]
    img2_coord = M[:, 2:]

    A = [[x*x_, x*y_, x, y*x_, y*y_, y, x_, y_, 1] for x,y,x_,y_ in zip(img1_coord[:,0], img1_coord[:,1], img2_coord[:,0], img2_coord[:,1])]
    A = np.array(A)

    u, s, vh = np.linalg.svd(A.T@A)

    F = (vh[-1,:]).reshape(3,3).T


    return F

def compute_T_matrix(xy, xy_):

    T = []

    for i in [xy, xy_]:
        T_center = np.array([[1, 0, -np.mean(i, axis=0)[0]],
                              [0, 1, -np.mean(i, axis=0)[1]],
                              [0, 0, 1]])

        i = np.array([T_center @ i_p for i_p in i])
        max_dis = np.max((i ** 2).sum(axis=1)) - 1
        max_dis = np.sqrt(max_dis)
        T_scaling = np.array([[1 / max_dis, 0, 0],
                              [0, 1 / max_dis, 0],
                              [0, 0, 1]])

        T.append(T_scaling @ T_center)

    return T

def compute_F_norm(M):

    img1_coord = M[:, :2].copy()
    img2_coord = M[:, 2:].copy()
    r = [[1]]*len(M)
    img1_coord = np.append(img1_coord, r, axis=1)
    img2_coord = np.append(img2_coord, r, axis=1)

    #T
    T = compute_T_matrix(img1_coord, img2_coord)

    T_img1_coord = (T[0]@img1_coord.T).T
    T_img2_coord = (T[1]@img2_coord.T).T

    T_img1_coord = T_img1_coord[:,:2]
    T_img2_coord = T_img2_coord[:,:2]


    M_norm = np.append(T_img1_coord, T_img2_coord, axis=1)

    M_norm = np.array(M_norm)
    F = compute_F_raw(M_norm)

    u, s, vh = np.linalg.svd(F)

    s_ = np.zeros((3,3))
    for i in range(2):
        s_[i,i] = s[i]
        
    F_ = np.dot(u, np.dot(s_,vh))

    F_ = np.dot(np.dot(T[1].T,F_), T[0])

    return F_


def compute_dis(M, F):
    img1_coord = M[:, :2].copy()
    img2_coord = M[:, 2:].copy()
    r = [[1]]*len(M)
    img1_coord = np.append(img1_coord, r, axis=1)
    img2_coord = np.append(img2_coord, r, axis=1)
    line1 = (F @ img1_coord.T).T
    line2 = (F.T @ img2_coord.T).T

    dis = 0

    for i in range(len(M)):
        Ax, Ay = map(int,[0, -line1[i,-1] / line1[i,1]])
        Bx = 1
        By = -(line1[i,0] * Bx + line1[i,-1]) / line1[i,1]
        area = abs ( (Ax - M[i, 2]) * (By - M[i,3]) - (Ay - M[i,3]) * (Bx - M[i,2]) )
        AB = ( (Ax - Bx) ** 2 + (Ay - By) ** 2 ) ** 0.5
        dis += (area / AB)**2
        Ax, Ay = map(int,[0, -line2[i,-1] / line2[i,1]])
        By = -(line2[i,0] * Bx + line2[i,-1]) / line2[i,1]
        area = abs ( (Ax - M[i, 0]) * (By - M[i,1]) - (Ay - M[i,1]) * (Bx - M[i,0]) )
        AB = ( (Ax - Bx) ** 2 + (Ay - By) ** 2 )**2
        dis += (area / AB) ** 2
    return dis



def compute_F_mine(M):
    start_time = time.time()
    seconds = 3
    brief_F = np.zeros((3,3))
    dis = 0
    while True:
        idx = list(range(len(M)))
        random.shuffle(idx)
        random_ = M[idx[:30]]
        F = compute_F_norm(random_)
        if np.all(brief_F == 0):
            brief_F = F
            dis = compute_dis(M,F)
        else:
            brief_dis = compute_dis(M,F)
            if dis > brief_dis:
                brief_F = F
                dis = brief_dis
        current_time = time.time()
        used_time = current_time - start_time
        if used_time >= seconds:
            break

    return brief_F

def draw_epi(F, img1, img2, p, q):
    r = [[1]]*len(p)
    p = np.append(p, r, axis=1)
    q = np.append(q, r, axis=1)
    y, x, z = img1.shape
    color = [(255,0,0), (0,255,0), (0,0,255)]
    line_ = (F @ p.T).T #img2에 나타내야됨
    line = (F.T @ q.T).T #img1에 나타내야됨
    img1_ = img1.copy()
    img2_ = img2.copy()

    for xyc, pn, qn, c in zip(line, p, q, color):
        Ax, Ay = map(int, [0,-xyc[-1]/xyc[1]])
        Bx, By = map(int,[x,-(xyc[-1]+xyc[0]*x)/xyc[1]])
        img1_ = cv2.line(img1_, (Ax,Ay),(Bx,By),c,1)
        img1_ = cv2.circle(img1_, (int(pn[0]), int(pn[1])), 3, c, 2)
        img2_ = cv2.circle(img2_, (int(qn[0]), int(qn[1])), 3, c, 2)

    for xyc, c in zip(line_, color):
        Ax, Ay = map(int, [0,-xyc[-1]/xyc[1]])
        Bx, By = map(int,[x,-(xyc[-1]+xyc[0]*x)/xyc[1]])
        img2_ = cv2.line(img2_, (Ax,Ay),(Bx,By),c,1)

    img = np.concatenate((img1_, img2_), axis=1)
    cv2.imshow('result', img)

    return




img1s = ['./temple1.png', './house1.jpg','./library1.jpg']
img2s = ['./temple2.png', './house2.jpg','./library2.jpg']
matches = ['temple_matches.txt', 'house_matches.txt', 'library_matches.txt']




for path1, path2, m in zip(img1s, img2s, matches):

    # 1-1
    M = np.loadtxt(m)

    print('Average Reprojection Errors (%s and %s)' % (path1[2:], path2[2:]))

    F_raw = compute_F_raw(M)
    error1 = error.compute_avg_reproj_error(M, F_raw)
    print('\tRaw =', error1)


    F_norm = compute_F_norm(M)
    error2 = error.compute_avg_reproj_error(M, F_norm)
    print('\tNorm =', error2)

    F_mine = compute_F_mine(M)
    error3 = error.compute_avg_reproj_error(M, F_mine)
    print('\tMine = ', error3, '\n')


    #1-2

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    key=0
    while True:
        if key == 113:
            cv2.destroyAllWindows()
            break
        else:
            idx = list(range(len(M)))
            random.shuffle(idx)
            random_3 = M[idx[:3]]
            draw_epi(F_mine, img1, img2, random_3[:,:2], random_3[:,2:])

        key = cv2.waitKey(0)
