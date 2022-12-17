import numpy as np
from scipy.optimize import linear_sum_assignment
dist_thres = 4

class KalmanFilter():
    def __init__(self, R_fac=10):
        self.R_fac = R_fac
        self.R = self.R_fac * np.eye(5)  # TODO 10 is the same setting as 2DsAP repo, considering the average inference time 2x, R_fac may be 4x=40
        self.Xt_1 = np.zeros([0, 5])  # [x, y, z, vx, vy]
        self.cls_prev = np.empty([0])
        self.P = np.empty([0, 5, 5])
        self.Q = np.eye(5)
        self.H = np.eye(5)
        self.F = np.eye(5)
        self.track_ratio = []

    def reset(self):
        self.R = self.R_fac * np.eye(5)
        self.Xt_1 = np.zeros([0, 5])
        self.cls_prev = np.empty([0])
        self.P = np.empty([0, 5, 5])
        self.Q = np.eye(5)
        self.H = np.eye(5)
        self.F = np.eye(5)

    def make_F(self, delta_time):
        self.F[[0, 1], [3, 4]] = delta_time

    def make_Q(self, delta_time):
        self.Q[[0,1,2], [0,1,2]] = delta_time**2
        self.Q[[3,4], [3,4]] = delta_time

    def predict(self, delta_time):
        N = self.Xt_1.shape[0]
        self.make_F(delta_time)
        F = self.F[None].repeat(N, 0)  # N 5, 5
        X_pred = (F @ self.Xt_1[:,:,None])[:,:,0]  # N 5
        self.make_Q(delta_time)
        Q = self.Q[None].repeat(N, 0)  # N 5 5
        self.P = F @ self.P @ F.transpose(0,2,1) + Q # [N, 5, 5]
        return X_pred

    def track(self, z, cls, x_pred):
        # z: [M, 5]
        # cls: [M]
        # x_pred: [N, 5]
        # cls_prev: [N, 1]
        M = z.shape[0]
        N = x_pred.shape[0]

        z_stack = z[:, None, :3].repeat(N, 1)  # [M, N, 3]
        x_pred_stack = x_pred[None, :, :3].repeat(M, 0)  # [M, N, 3]
        affinity_mat = np.linalg.norm(z_stack - x_pred_stack, axis=2)  # [M, N]

        cls_stack = cls[:, None].repeat(N, 1)  # [M, N]
        cls_prev_stack = self.cls_prev[None, :].repeat(M, 0)  # [M, N]
        affinity_mask = (cls_stack == cls_prev_stack)  # [M, N]
        self.cls_prev = cls

        affinity_mat[~affinity_mask] = 1e6
        idx1, idx2 = linear_sum_assignment(affinity_mat)
        keep_idx1 = []
        keep_idx2 = []
        for i1, i2 in zip(idx1, idx2):
            if affinity_mat[i1, i2] < dist_thres:
                keep_idx1.append(i1)
                keep_idx2.append(i2)

        keep_idx1 = np.array(keep_idx1)
        keep_idx2 = np.array(keep_idx2)

        if M:
            self.track_ratio.append(len(keep_idx1) / M)

        return keep_idx1, keep_idx2

    def update(self, z, x_pred, idx1, idx2):
        # idx1是给当前的观测用的
        # idx2是给上一时刻用的
        N2 = len(idx1)
        M = z.shape[0]

        if N2:
            # update
            H = self.H[None].repeat(N2, 0)  # N2 5 5
            y = z[idx1] - (H @ x_pred[idx2][:,:,None])[:,:,0]  # [N2, 5]
            R = self.R[None].repeat(N2, 0)  # N2 5 5
            S = H @ self.P[idx2] @ H.transpose(0,2,1) + R # [N2, 5, 5]
            K = self.P[idx2] @ H.transpose(0,2,1) @ np.linalg.inv(S)  # [N2, 5, 5]
            P_tmp = (np.eye(5)[None].repeat(N2, 0) - K @ H) @ self.P[idx2] # [N2, 5, 5]

            x_update = x_pred[idx2] + (K @ y[:,:,None])[:,:,0] # [N2, 5]
            self.Xt_1 = np.zeros([M, 5])
            self.Xt_1[:,:5] = z  # [M, 5]
            self.Xt_1[idx1] = x_update
            self.P = self.R_fac * np.eye(5)[None].repeat(M, 0) # [M, 5, 5]
            self.P[idx1] = P_tmp
        else:
            self.Xt_1 = z
            self.P = self.R_fac * (np.eye(5)[None]).repeat(M, 0) # [M, 5, 5]


    def __call__(self, Zt, cls, delta_time):
        # Zt: [x, y, z, vx, vy]是该时刻的观测 [M,5]
        # cls: [M] label
        # delta_time: 两帧之间的时间间隔

        # 1. predict
        X_pred = self.predict(delta_time) if len(self.Xt_1) else np.empty([0, 5])  # N 5

        # 2. track
        idx1, idx2 = self.track(Zt, cls, X_pred)  # [M]

        # 3. update
        self.update(Zt, X_pred, idx1, idx2)

        return self.Xt_1  # [M, 5]