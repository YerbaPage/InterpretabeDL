import torch
from torch import nn
from torch.nn import functional as F
import numpy as NP
import math
from Config_File import *
from torch.nn.parameter import Parameter
from modules import MultiHeadAttention

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(6*in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.WQ = nn.Linear(in_features, in_features)
        self.fc=nn.Linear(in_features*3,out_features)

    def forward(self, input, adj, ax=None, adj_label=None): #input: B*n*dim, adj: B*n*n
        input=input.squeeze().permute(0,2,1)
        if ax is not None:
            ax = ax.view(ax.shape[0],-1,ax.shape[-1]).permute(0, 2, 1)
        h = self.WQ(input)
        B,N,D=h.shape

        if ax is not None:  # q,k,v: (Batch,nhead * head_dim,Len,1)
            h = torch.cat([h, ax], -1)


        a_input = torch.cat([h.repeat(1,1, N).view(B,N * N,-1), h.repeat(1,N, 1)], dim=-1).view(B,N, N, -1)  #B N N 2*dim
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # B N N

        zero_vec = -9e15*torch.ones_like(e)  #B N N
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  #B N N
        attention = F.dropout(attention, self.dropout)  #B N N
        h_prime = torch.bmm(attention, h)  #B N dim

        h_prime=self.fc(h_prime)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DependencyGNN_GAT(nn.Module):
    """Star-Transformer Encoder part。
    paper: https://arxiv.org/abs/1902.09113

    :param hidden_size: int, 输入维度的大小。同时也是输出维度的大小。
    :param num_layers: int, star-transformer的层数
    :param num_head: int，head的数量。
    :param head_dim: int, 每个head的维度大小。
    :param dropout: float dropout 概率
    :param max_len: int or None, 如果为int，输入序列的最大长度，
                    模型会为属于序列加上position embedding。
                    若为None，忽略加上position embedding的步骤
    """
    def __init__(self, hidden_size, num_layers, num_head, head_dim, dropout=0.1, max_len=None):
        super(DependencyGNN_GAT, self).__init__()
        self.iters = num_layers
        self.num_head=num_head
        self.d_model=hidden_size

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.iters)])
        self.ring_att = nn.ModuleList([GraphAttentionLayer(hidden_size,head_dim,dropout=dropout,alpha=0.2) for _ in range(num_head)])
        self.star_att = nn.ModuleList(
            [MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=dropout)
                for _ in range(self.iters)])

        if max_len is not None:
            self.pos_emb = nn.Embedding(max_len, hidden_size)
        else:
            self.pos_emb = None

        self.fc=nn.Linear(num_head*head_dim,hidden_size)
        self.fc.to(Config.device)


    def forward(self, data, mask, edge_matrix):
        """
        :param FloatTensor data: [batch, length, hidden] the input sequence
        :param ByteTensor mask: [batch, length] the padding mask for input, in which padding pos is 0
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        """
        def norm_func(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        B, L, H = data.size()
        mask=(mask==0)
        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = data.permute(0, 2, 1)[:,:,:,None] # B H L 1
        if self.pos_emb:
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device)\
                    .view(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
            embs = embs + P

        nodes = embs
        relay = embs.mean(2, keepdim=True)
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        r_embs = embs.view(B, H, 1, L)
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)
            temp=torch.cat([self.ring_att[j](norm_func(self.norm[i], nodes), adj=edge_matrix,ax=ax) for j in range(self.num_head)],-1)
            temp=self.fc(temp)
            nodes = nodes + F.leaky_relu(temp).permute(0,2,1).unsqueeze(-1)
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))

            nodes = nodes.masked_fill_(ex_mask.byte(), 0)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)

        return nodes, relay.view(B, H)


class MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim*3, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 5

        self.weight_k = Parameter(torch.FloatTensor(head_dim, head_dim))
        nn.init.xavier_normal_(self.weight_k)
        self.weight_k.to(Config.device)
        self.weight_v = Parameter(torch.FloatTensor(head_dim, head_dim))
        nn.init.xavier_normal_(self.weight_v)
        self.weight_v.to(Config.device)

        self.bias_k = Parameter(torch.FloatTensor(1,head_dim))
        nn.init.xavier_normal_(self.bias_k)
        self.bias_k.to(Config.device)

        self.bias_v = Parameter(torch.FloatTensor(1,head_dim))
        nn.init.xavier_normal_(self.bias_v)
        self.bias_v.to(Config.device)


    def forward(self, x, ax=None, edge_matrix=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,n_head*H,L,1)   q,k,v: (Batch,nhead * head_dim,Len,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = q.view(B, nhead, head_dim, 1, L)
        v = q.view(B, nhead, head_dim, 1, L)

        edge_matrix=edge_matrix.unsqueeze(1).expand(B,nhead,L,L).contiguous().view(B*nhead,L,L)

        '''k=k.view(B,nhead,head_dim,L).permute(0,1,3,2).view(B*nhead,L,head_dim) # B*nhead, L, hidden_dim
        k_support=torch.bmm(k, self.weight_k.unsqueeze(0).expand(B*nhead,head_dim,head_dim))
        k = torch.bmm(edge_matrix, k_support) + self.bias_k.unsqueeze(0).expand(B*nhead,L,head_dim)  # B*nhead, L, head_dim
        k=F.leaky_relu(k.permute(0,2,1).view(B, nhead, head_dim, 1, L))

        v=v.view(B,nhead,head_dim,L).permute(0,1,3,2).view(B*nhead,L,head_dim) # B*nhead, L, hidden_dim
        v_support=torch.bmm(v, self.weight_v.unsqueeze(0).expand(B*nhead,head_dim,head_dim))
        v = torch.bmm(edge_matrix, v_support) + self.bias_v.unsqueeze(0).expand(B*nhead,L,head_dim)  # B*nhead, L, head_dim
        v=F.leaky_relu(v.permute(0,2,1).view(B, nhead, head_dim, 1, L))'''

        '''k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0))\
                .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0))\
                .view(B, nhead, head_dim, unfold_size, L)'''
        if ax is not None:  # q,k,v: (Batch,nhead * head_dim,Len,1)
            k = torch.cat([k, ak], 3)  # B, nhead, head_dim, 1+ak_size, L
            v = torch.cat([v, av], 3)  # B, nhead, head_dim, 1+ak_size, L

        k=k.view(B,nhead,-1,L).permute(0,1,3,2)  # B, nhead, L, dim*(1+ak_size)
        v=v.view(B,nhead,-1,L).permute(0,1,3,2)  # B, nhead, L, dim*(1+ak_size)
        dim=v.shape[-1]

        alpha=torch.bmm(k.view(B*nhead,L,-1),v.view(B*nhead,L,-1).permute(0,2,1)) # B*nhead, L, L
        alpha=F.softmax(alpha*edge_matrix.view(B*nhead,L,L)/ NP.sqrt(head_dim), 2)
        alpha=self.drop(alpha)  # B*nhead, L, L
        att = torch.bmm(alpha,v.view(B*nhead,L,-1)).view(B, -1, L, 1) # B, nhead*(dim*(1+ak_size)) , L, 1

        #alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  # B N L 1 U
        #att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)
        #att=v.view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)

        return ret


class MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(MSA2, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = k.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / NP.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :].byte(), -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)
