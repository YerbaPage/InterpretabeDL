import torch
from torch import nn
from torch.nn import functional as F
import numpy as NP
import math
from Config_File import *
from torch.nn.parameter import Parameter
from modules import MultiHeadAttention
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_layer, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layer=num_layer
        self.d_model=out_features
        self.weights = [Parameter(torch.FloatTensor(in_features if i==0 else out_features, out_features).to(Config.device)) for i in range(self.num_layer)]
        self.d_model=out_features
        if bias:
            self.bias = [Parameter(torch.FloatTensor(out_features).to(Config.device)) for _ in range(self.num_layer)]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.layer_norm = nn.LayerNorm(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights[0].size(1))
        for i in range(self.num_layer):
            self.weights[i].data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input, adj, input_mask=None): # input: batch * len * dim, adj: bath * len * len
        for i in range(self.num_layer):
            support = torch.matmul(input, self.weights[i])  #input: batch * len * out_dim
            output = torch.matmul(adj, support) #
            if self.bias is not None:
                output=output + self.bias[i]
            if input_mask is not None:
                output=output*input_mask.unsqueeze(-1).float()
            input=self.layer_norm(F.relu(output) + input)
        return F.relu(input), F.relu(input[:,0,:].squeeze())

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DependencyGNN(nn.Module):
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
        super(DependencyGNN, self).__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.iters)])
        self.ring_att = nn.ModuleList(
            [MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=dropout,temperature=math.sqrt(i+1))
                for i in range(self.iters)])
        self.star_att = nn.ModuleList(
            [MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=dropout)
                for _ in range(self.iters)])
        self.d_model=hidden_size

        '''self.ring_att = nn.ModuleList(
            [MultiHeadAttention(n_head=num_head,d_model=hidden_size,d_k=64, d_v=64,dropout=dropout)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [MultiHeadAttention(n_head=num_head, d_model=hidden_size, d_k=64, d_v=64, dropout=dropout)
             for _ in range(self.iters)])'''

        if max_len is not None:
            self.pos_emb = self.pos_emb = nn.Embedding(max_len, hidden_size)
        else:
            self.pos_emb = None


    def forward(self, data, mask, edge_matrix,edge_label):
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
            res_nodes=nodes
            res_relay=relay
            if i==0:
                edge_matrix_input=edge_matrix
            else:
                edge_matrix_input = edge_matrix
                #attn_max=torch.max(attn,-1)
                #edge_matrix_input=torch.where(attn>attn_max*0.3,1,0)
            nodes, attn = self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax, edge_matrix=edge_matrix_input, edge_label=edge_label)
            nodes = F.leaky_relu(nodes)
            nodes = nodes.masked_fill_(ex_mask.byte(), 0)
            nodes=norm_func(self.norm[i],res_nodes+nodes)

            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))

            relay = norm_func(self.norm[i], res_relay+relay)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)

        return nodes, relay.view(B, H)


class MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1, temperature=1.0):
        super(MSA1, self).__init__()
        self.temperature=temperature
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WE = nn.Linear(Config.edge_label_dim, nhead * head_dim)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

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

        self.edge_emb_layer=nn.Embedding(Config.max_edge_label_count, head_dim)

        self.WEDGE= nn.Linear(2*head_dim,1)
        #nn.init.xavier_normal_(self.bias_v)


    def forward(self, x, ax=None, edge_matrix=None, edge_label=None):
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

        edge_matrix=edge_matrix.unsqueeze(1).expand(B,nhead,L,L).contiguous().view(B,nhead,L,L)


        '''if ax is not None:  # q,k,v: (Batch,nhead * head_dim,Len,1)
            k = torch.cat([k, ak], 3)  # B, nhead, head_dim, 1+ak_size, L
            v = torch.cat([v, av], 3)  # B, nhead, head_dim, 1+ak_size, L'''

        if edge_label is not None:
            edge_emb = self.edge_emb_layer(edge_label)  # B L L H
            edge_emb=self.WE(edge_emb) # B L L H*head
            edge_emb=edge_emb.view(B,L,L,nhead,head_dim).permute(0,3,1,2,4) # B NHEAD L L H

        k=k.view(B,nhead,-1,L).permute(0,1,3,2)  # B, nhead, L, dim*(1+ak_size)
        v=v.view(B,nhead,-1,L).permute(0,1,3,2)  # B, nhead, L, dim*(1+ak_size)
        dim=v.shape[-1]

        a_input = torch.cat([k.repeat(1, 1, 1, L).view(B,nhead, L * L, dim), k.repeat(1, 1, L, 1).view(B,nhead, L * L, dim)], dim=-1).view(B, nhead, L,
                                                                                                                   L,
                                                                                                                   2 * dim)  # B nhead N N 2*dim
        if edge_label is not None:
            a_input=torch.cat([a_input,edge_emb],-1)
        alpha=self.WEDGE(a_input).squeeze() # B nhead L L

        #alpha=torch.bmm(k.view(B*nhead,L,-1),v.view(B*nhead,L,-1).permute(0,2,1)) # B*nhead, L, L  这种方法没有引入额外参数，但是GAT引入了
        alpha=alpha.view(B*nhead,L,L)/ NP.sqrt(head_dim)
        if edge_matrix is not None:
            alpha=alpha.masked_fill_(1-edge_matrix.view(B*nhead,L,L).byte(), -1e15)
        alpha=F.softmax(alpha, 2)
        alpha=alpha/self.temperature

        alpha=self.drop(alpha)  # B*nhead, L, L
        att = torch.bmm(alpha,v.view(B*nhead,L,-1)).view(B, -1, L, 1) # B, nhead*(dim*(1+ak_size)) , L, 1

        #alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  # B N L 1 U
        #att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)
        #att=v.view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)

        return ret, alpha


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
