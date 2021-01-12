import torch
from torch import nn
from torch.nn import functional as F
import numpy as NP
from modules import MultiHeadAttention
from Transformer import Transformer
#from torch_geometric.nn import GINConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
#from Transformer_compositional import Transofmer_compositional

class Transformer_with_graph(Transformer):
    def __init__(
            self,
            n_src_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, gnn="GCNConv", n_graph_layer=1
    ):
        super().__init__(n_src_vocab,len_max_seq,d_word_vec,d_model,d_inner,n_layers,n_head,d_k,d_v,dropout)
        mlp = Seq(Lin(d_word_vec, d_inner), ReLU(), Lin(d_inner, d_model), ReLU())
        #self.gnn=nn.ModuleList([GINConv(mlp) for i in range(n_graph_layer)])
        self.gnn = nn.ModuleList([Transofmer_compositional(n_src_vocab, len_max_seq,
            d_word_vec, d_model, d_inner,
            n_layers, n_head, d_k, d_v) for i in range(n_graph_layer)])
        self.n_graph_layer=n_graph_layer
        self.fc_merge=nn.Linear(d_model*2, d_model)
        nn.init.xavier_normal_(self.fc_merge.weight)
        self.layer_norm_merge = nn.LayerNorm(d_model)
        self.dropout_merge=nn.Dropout(dropout)

    def forward(self, src_seq, src_embed, src_pos, edge,root):
        residual=src_embed
        enc_output, *_ = self.encoder(src_seq, src_embed, src_pos)

        enc_graph,_ = self.gnn[0](src_seq, src_embed, src_pos)
        '''enc_graph=src_embed
        for i_layer in range(self.n_graph_layer):
            enc_graph_new=[]
            for i in range(len(src_pos)):
                edge_i=edge[i][:torch.sum(src_pos[i,:])*2].permute(1,0)
                #enc_graph_new.append(self.gnn[i_layer](enc_graph[i],edge_i))
            enc_graph=torch.cat(enc_graph_new,0).reshape(enc_output.size())
        #enc_graph = self.gnn(src_embed,edge[:torch.sum(src_pos.byte(),1)])'''


        #enc_output=self.dropout_merge(self.fc_merge(torch.cat((enc_output,enc_graph),-1)))
        enc_output=self.layer_norm_merge(residual+enc_output+enc_graph)
        return enc_output,enc_output[:, 0, :].squeeze()

        return enc_output, torch.gather(enc_output,-2,root.unsqueeze(-1).expand(-1,-1,enc_output.size()[-1])).squeeze() #enc_output[:, 0, :].squeeze()

class StarTransformer(nn.Module):
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
        super(StarTransformer, self).__init__()
        self.iters = num_layers

        self.d_model=hidden_size

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.iters)])
        self.ring_att = nn.ModuleList(
            [MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=dropout)
                for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=dropout)
                for _ in range(self.iters)])

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

    def forward(self, data, mask):
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
        mask = (mask == 0)
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
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)  # Batch  HiddenDim  2  Len
            nodes_ex=nodes
            nodes = nodes + F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))   # Batch HiddenDim Len 1
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))      # Batch HiddenDim 1 1

            nodes = nodes.masked_fill_(ex_mask.bool(), 0)

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
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 5

    def forward(self, x, ax=None): # x: (Batch,HiddenDim,Len,1)  ax: Batch  HiddenDim  2  Len
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (Batch,HiddenDim,Len,1) q,k,v: (Batch,nhead * head_dim,Len,1)

        if ax is not None:
            aL = ax.shape[2]  # 2
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)  # B  nhead, head_dim, 1, L
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0))\
                .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0))\
                .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)  #B, nhead, head_dim, unfold_size+2, L
            v = torch.cat([v, av], 3)  #B, nhead, head_dim, unfold_size+2, L

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  #B, nhead, head_dim, unfold_size+2, L
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

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
            pre_a = pre_a.masked_fill(mask[:, None, None, :].bool(), -10000000000) #-float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)
