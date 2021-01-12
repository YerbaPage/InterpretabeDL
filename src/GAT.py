import torch
from torch import nn
from torch.nn import functional as F
from modules import MultiHeadAttention
from GraphNN import MSA2
from Config_File import Config

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
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, adj_label=None): #input: B*n*dim, adj: B*n*n
        h = torch.matmul(input, self.W) #B*n*dim
        B,N=h.shape[0],h.shape[1]

        a_input = torch.cat([h.repeat(1,1, N).view(B,N * N, self.out_features), h.repeat(1,N, 1)], dim=-1).view(B,N, N, 2 * self.out_features)  #B N N 2*dim
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # B N N

        zero_vec = -9e15*torch.ones_like(e)  #B N N
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  #B N N
        attention = F.dropout(attention, self.dropout)  #B N N
        h_prime = torch.bmm(attention, h)  #B N dim

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers, max_len=None):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nclass=nclass
        self.nlayers=nlayers

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.self_att=MultiHeadAttention(nheads,nclass,nclass,nclass)
        self.star_att=MSA2(nfeat, nhead=nheads, head_dim=nhid, dropout=dropout)

        self.norm=nn.LayerNorm(nclass)

        self.d_model=nclass

        if max_len is not None:
            self.pos_emb = self.pos_emb = nn.Embedding(max_len, nfeat)
        else:
            self.pos_emb = None

    def forward(self, x, adj,mask):

        mask=(mask==0)
        if self.pos_emb is not None:
            L = x.shape[1]
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=x.device)\
                    .view(1, L))  # 1 L H
            x = x + P

        #x = F.dropout(x, self.dropout)
        for i in range(self.nlayers):
            y = F.leaky_relu(torch.cat([att(x, adj) for att in self.attentions], dim=2))
            y = F.dropout(y, self.dropout, training=self.training)
            x = x+ F.elu(self.out_att(y, adj))   #B N dim
            x = x.masked_fill_(mask.unsqueeze(-1).expand(-1,-1,self.nclass).byte(), 0)
            x=self.norm(x)


        #x=F.log_softmax(x, dim=1)
        #x,_=self.self_att(x,x,x,mask=mask.unsqueeze(1).expand(-1, x.shape[1], -1))
        nodes=x.permute(0, 2, 1)[:,:,:,None]
        relay = nodes.mean(2, keepdim=True)
        smask = torch.cat([torch.zeros(x.shape[0], 1, ).byte().to(mask), mask], 1)
        #return F.leaky_relu(self.star_att(relay, torch.cat([relay, nodes], 2), smask)).squeeze()

        return F.relu(x), F.relu(x[:,0,:]).squeeze()
        #return torch.mean(torch.stack(ret))

class GAT_neibhgor(GAT):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers, length2):
        """Dense version of GAT."""
        super(GAT_neibhgor, self).__init__(nfeat, nhid, nclass, dropout, alpha, nheads, nlayers)

        self.pos_emb = None
        self.adj=torch.eye(length2).unsqueeze(0).to(Config.device)
        length=int(length2/2)
        for i in range(length):
            self.adj[0,i,i+length]=1
            self.adj[0,i+length,i]=1
            if i>0:
                self.adj[0,i-1,i+length]=1
                self.adj[0,i+length,i-1]=1


    def forward(self, x, mask):
        return super(GAT_neibhgor, self).forward(x,self.adj,mask)
