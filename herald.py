import torch
import torch.nn as nn
import math

import torch.nn.functional as F


class Generate_G_from_H(nn.Module):
    def __init__(self):
        super(Generate_G_from_H, self).__init__()

    def forward(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        n_edge = H.shape[1]  # 4024
        # the weight of the hyperedge
        W = torch.ones(n_edge).type_as(H)  
        # the degree of the node
        DV = torch.sum(H * W,
                       dim=1)
        # the degree of the hyperedge
        DE = torch.sum(H, dim=0)  # [4024]


        invDE = torch.diag(torch.pow(DE, -1))
        invDE[torch.isinf(invDE)] = 0  # D_e ^-1
        invDV = torch.pow(DV, -0.5)
        invDV[torch.isinf(invDV)] = 0
        DV2 = torch.diag(invDV)  # D_v^-1/2

        W = torch.diag(W)
        # H = np.mat(H)
        HT = H.t()

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2.mm(H).mm(W).mm(invDE).mm(HT).mm(DV2)
            # G = DV2 * H * W * invDE * HT * DV2  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
            return G



class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query):
        scores = torch.matmul(query, query.transpose(0, 1)) / math.sqrt(query.shape[1])  # Q*V.T/sqrt(d_k)
        return scores


class HERALD(nn.Module):
    """
    HERALD Module
    """

    def __init__(self, in_feature, hidden, only_G=True, theta=0.01):
        """
        :param dim_in: input feature dimension
        """
        super(HERALD, self).__init__()

        self.linear = nn.Linear(in_feature, hidden, bias=False)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(hidden, hidden))
        self.W_v = nn.Linear(in_feature, hidden, bias=False)  # bidirectional
        self.w_o = nn.Linear(hidden, 1)
        self._generate_G_from_H = Generate_G_from_H()
        self.use_binarize = False
        self.only_G = only_G
        self.theta = theta
        self.softmax = nn.Softmax(dim=1)
        self.Scales = ScaledDotProductAttention()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)


    def hadamard_power(self, x, y):
        scores = []
        N = y.shape[0]
        M = x.size(0)
        x = x.repeat(1, N).view(M * N, -1)  #
        y = y.repeat(M, 1)
        dist = (x - y) ** 2

        scores = self.w_o(dist).view(M, N)
        return scores

    def forward(self, adj=None, G=None, feats=None, kn=10, num=None,sigma=20):
        """
        :param adj: (v_n, e_n)
        :param  G: (v_n, v_n) Generate_G_from_H(adj)
        :return:G_new (v_n, v_n)
        """
        adj = adj.t()  # (e_n, v_n)
        ## 1. Learnable Incidence Matrix

        # hyperedge feature
        e_center = feats.t().mm(adj.t()) / adj.sum(1)
        s = self.linear(e_center.t())
        # attn
        feats = self.W_v(feats)

        scores = self.Scales(feats)
        attn = self.softmax(scores)


        d = torch.matmul(attn, feats)

        s = self.layer_norm1(s)
        d = self.layer_norm2(d)
        transformed_adj = self.hadamard_power(s, d)
        transformed_adj = torch.exp(-transformed_adj / (2 * sigma ** 2))

        G_new = self._generate_G_from_H(transformed_adj.t())
        self.adj = G_new


        ##  2. Residual Dynamic Laplacia
        if num is not None:
            theta = 1 - (1 - self.theta) * (math.cos(math.pi * (num - 1) / 10) + 1) / 2
        else:
            theta = self.theta

        G_new = (1 - theta) * G + theta * G_new

        if self.only_G:
            return G_new.float()  # x, attn
        return torch.spmm(G_new, feats)  # nearest_feature



if __name__ == '__main__':
    import numpy as np

    initial_H = torch.randn(5, 8) > 0.
    initial_H = initial_H.type(torch.float)
    # dist = np.random(5, 5)
    generate_G_from_H = Generate_G_from_H()
    G = generate_G_from_H(initial_H)

    x = torch.randn(5, 16)
    # N = I - L
    Herabd =  HERALD(in_feature=16, hidden=4,)
    Learnable_G =Herabd(adj=initial_H,G=G, feats=x)

    print('learnable laplacian is',torch.eye(Learnable_G.shape[0])-Learnable_G)

