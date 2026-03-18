import numpy as np
class KalmanFilterSimple:

    def __init__(self):

        # 状态向量 [cx, cy, w, h, vx, vy, vw]
        self.x = np.zeros((7,1))
        dt=1
        # 状态转移矩阵
        self.F = np.array([
            [1,0,0,0,dt,0,0],
            [0,1,0,0,0,dt,0],
            [0,0,1,0,0,0,dt],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])

        # 观测矩阵
        self.H = np.zeros((4,7))
        self.H[:4,:4] = np.eye(4)

        # 协方差
        self.P = np.eye(7) * 10

        # 噪声
        self.Q = np.eye(7) * 0.01
        self.R = np.eye(4) * 1


    def predict(self):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q


    def update(self,z):

        z = z.reshape((4,1))

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])

        self.P = (I - K @ self.H) @ self.P