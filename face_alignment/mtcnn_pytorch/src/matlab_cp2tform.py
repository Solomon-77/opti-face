import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank

class MatlabCp2tormException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
                __file__, super.__str__(self))

def tformfwd(trans, uv):
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy

def tforminv(trans, uv):
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy

def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv

def findSimilarity(uv, xy, options=None):
    options = {'K': 2}

    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    trans2 = np.dot(trans2r, TreflectY)

    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv

def get_similarity_transform(src_pts, dst_pts, reflective=True):
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)

    return trans, trans_inv

def cvt_tform_mat_for_cv2(trans):
    cv2_trans = trans[:, 0:2].T
    return cv2_trans

def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)
    return cv2_trans

if __name__ == '__main__':
    u = [0, 6, -2]
    v = [0, 3, 5]
    x = [-1, 0, 4]
    y = [-1, -10, 4]

    uv = np.array((u, v)).T
    xy = np.array((x, y)).T

    print('\n--->uv:')
    print(uv)
    print('\n--->xy:')
    print(xy)

    trans, trans_inv = get_similarity_transform(uv, xy)

    print('\n--->trans matrix:')
    print(trans)

    print('\n--->trans_inv matrix:')
    print(trans_inv)

    print('\n---> apply transform to uv')
    print('\nxy_m = uv_augmented * trans')
    uv_aug = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy_m = np.dot(uv_aug, trans)
    print(xy_m)

    print('\nxy_m = tformfwd(trans, uv)')
    xy_m = tformfwd(trans, uv)
    print(xy_m)

    print('\n---> apply inverse transform to xy')
    print('\nuv_m = xy_augmented * trans_inv')
    xy_aug = np.hstack((
        xy, np.ones((xy.shape[0], 1))
    ))
    uv_m = np.dot(xy_aug, trans_inv)
    print(uv_m)

    print('\nuv_m = tformfwd(trans_inv, xy)')
    uv_m = tformfwd(trans_inv, xy)
    print(uv_m)

    uv_m = tforminv(trans, xy)
    print('\nuv_m = tforminv(trans, xy)')
    print(uv_m)