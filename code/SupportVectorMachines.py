import numpy as np
from cvxopt import matrix, solvers


class SupportVectorMachines:
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y
        self.d, self.n = np.shape(x)
        self.X_mat = None
        self.P = None
        self.G = None
        self.h = None
        self.alpha = None
        self.w = None
        self.b = None

    def train(self, C: float=float("inf"), core=np.dot):
        self.initial_mat(C, core)
        P = matrix(self.P.T)
        q = matrix(-np.ones(self.n))
        G = matrix(self.G.T)
        h = matrix(self.h)
        A = matrix(self.y)
        b = matrix(np.zeros(1))
        result = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        self.alpha = np.array(result['x']).T
        self.w = np.dot(x*y, self.alpha.T)
        for i in range(self.n):
            if 0 < self.alpha[0, i] < C:
                self.b = self.y[0, i] -  np.dot(self.y, (self.alpha*self.X_mat[:, i]).T)
                break
        return self.w, self.b

    def initial_mat(self, C: float, core):

        if core == np.dot:
            self.X_mat = np.dot(self.x.T, self.x)
        else:
            self.X_mat = np.array([[core(self.x[:, i], self.x[:, j]) for j in range(self.n)] for i in range(self.n)])
        self.P = np.dot(self.y.T, self.y) * self.X_mat
        self.P = np.array(self.P, dtype=float)
        #self.P = np.array([[self.P[i, j] / 2 if i != j else self.P[i, j]
        #                        for j in range(self.n)] for i in range(self.n)])

        if C == float("inf"):
            self.G = -np.eye(self.n)
            self.h = np.zeros(self.n)
        else:
            self.G = np.append(-np.eye(self.n), np.eye(self.n), axis=0)
            self.h = np.append(np.zeros(self.n), np.ones(self.n)*C)

if __name__ == "__main__":
    x = np.array([[3, 4, 1], [3, 3, 1]])
    y = np.array([[1, 1, -1]], dtype=float)
    supportVectorMachines = SupportVectorMachines(x, y)
    print(supportVectorMachines.train())

a = {"ajlx":"2001",
     "ajlxmc":"一审公诉案件",
     "ajmc":"张瑞涛涉嫌危险驾驶案",
     "bmsah":"沭检刑诉受[2021]371329000004号",
     "dqczsj":1625015064234,"dwbm":"371329",
     "logTimestamp":0,
     "rjrbh":"3713290011",
     "rjrq":1625015064234,
     "rjrxm":"樊海洋",
     "tysah":"37132920210000420",
     "wslb":[
         {"glzrrbh":"",
          "wsmbbh":"100000030268",
          "wsmc":"送达回证",
          "wsslbh":"72E9C53AC59B4C3DA77B8CF5A30D0D40"
          },
         {"glzrrbh":"",
          "wh":"沭检速建〔2021〕1号",
          "wsmbbh":"100000990364",
          "wsmc":"适用速裁程序建议书",
          "wsslbh":"A596A087B295496DBC914026526A2DC5"
          },
         {"glzrrbh":"",
          "wh":"沭检量建〔2021〕1号",
          "wsmbbh":"100000030180",
          "wsmc":"量刑建议书",
          "wsslbh":"D38D5EFF1C8C49C6B911D929FF97539E"
          }
     ],
     "yxslbh":"沭检刑诉受[2021]371329000004号"}
