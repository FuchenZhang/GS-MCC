import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math
from torch.autograd import Variable

class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)
        return loss

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss

class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked/alpha_sum
        else:
            M_ = M.transpose(0,1)
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1)
            M_x_ = torch.cat([M_,x_],2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2)

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper.
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        edge_idn -> edge_idn是边的index的集合
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            if not self.no_cuda:
                mask = (torch.ones(alpha.size()) * 1e-10).detach().cuda()
                mask_copy = torch.zeros(alpha.size()).detach().cuda()

            else:
                mask = (torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = torch.zeros(alpha.size()).detach()

            edge_ind_ = []
            for i, j in enumerate(edge_ind):
                for x in j:
                    edge_ind_.append([i, x[0], x[1]])

            edge_ind_ = np.array(edge_ind_).transpose()
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy

            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):
    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def simple_batch_graphify(features, lengths, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features, None, None, None, None


def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))
    features = features.permute(1, 0, 2)
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:, j, :])

        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()

            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """

    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()

    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda)
                            for s, l in zip(start.data.tolist(),
                                            input_conversation_length.data.tolist())], 0).transpose(0, 1)

    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:, 0, :])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer,
                           nodal_attn, avec, no_cuda):
    if nodal_attn:

        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob

class FGN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x):

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, x):
        B, N = x.shape
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x)

        x = x + bias

        x = x.reshape(B, (N)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")

        x = x.reshape(B, N, self.embed_size)

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, -1)
        x = self.fc(x)

        return x


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, hidden_size=64, dropout=0.5,
                 no_cuda=False, return_feature=False):
        super(GraphNetwork, self).__init__()

        self.return_feature = return_feature
        self.no_cuda = no_cuda
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = FGN(hidden_size, hidden_size, hidden_size, hidden_size, hidden_size)
        if not self.return_feature:
            self.matchatt = MatchingAttention(num_features + hidden_size, num_features + hidden_size,
                                              att_type='general2')
            self.linear = nn.Linear(num_features + hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.smax_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        out = self.conv1(x, edge_index, edge_type)
        out = self.conv2(out)
        emotions = torch.cat([x, out], dim=-1)
        if self.return_feature:
            return emotions
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout,
                                          self.smax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general'):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)
        if att_type == 'av_bg_fusion':
            self.transform_al = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type == 'general':
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim * 3, 1)
            self.transform_al = nn.Linear(mem_dim * 3, 1)
            self.transform_vl = nn.Linear(mem_dim * 3, 1)

    def forward(self, a, v, l, modals=None):
        a = self.dropouta(a) if len(a) != 0 else a
        v = self.dropoutv(v) if len(v) != 0 else v
        l = self.dropoutl(l) if len(l) != 0 else l
        if self.att_type == 'av_bg_fusion':
            if 'a' in modals:
                fal = torch.cat([a, l], dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa * (self.scalar_al(a))
            if 'v' in modals:
                fvl = torch.cat([v, l], dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv * (self.scalar_vl(v))
            if len(modals) == 3:
                hmf = torch.cat([l, hma, hmv], dim=-1)
            elif 'a' in modals:
                hmf = torch.cat([l, hma], dim=-1)
            elif 'v' in modals:
                hmf = torch.cat([l, hmv], dim=-1)
            return hmf
        elif self.att_type == 'general':
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a * v], dim=-1)))
                h_av = z_av * ha + (1 - z_av) * hv
                if 'l' not in modals:
                    return h_av
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a, l, a * l], dim=-1)))
                h_al = z_al * ha + (1 - z_al) * hl
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v, l, v * l], dim=-1)))
                h_vl = z_vl * hv + (1 - z_vl) * hl
                if 'a' not in modals:
                    return h_vl
            return torch.cat([h_av, h_al, h_vl], dim=-1)


class DialogueGCNModel(nn.Module):

    def __init__(self, base_model, D_m, D_e, D_m_v, D_m_a, graph_hidden_size, n_speakers, max_seq_len,
                 window_past, window_future,
                 n_classes=7, dropout=0.5,
                 nodal_attention=True, avec=False,
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6,
                 graph_construct='direct', use_residue=True,
                 dynamic_edge_w=False, hidden_dim=200, modals='avl', att_type='gated', av_using_lstm=False,
                dataset='IEMOCAP',
                 use_speaker=True, use_modal=False):

        super(DialogueGCNModel, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type = graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.n_speakers = n_speakers
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx)
        self.att_type = att_type
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = dataset


        self.linear_a = nn.Linear(D_m_a, hidden_dim)
        self.linear_v = nn.Linear(D_m_v, hidden_dim)
        self.linear_t = nn.Linear(D_m, hidden_dim)

        self.lstm_t = nn.LSTM(input_size=hidden_dim, hidden_size=D_e, num_layers=2, bidirectional=True,
                            dropout=dropout)
        self.lstm_a = nn.LSTM(input_size=hidden_dim, hidden_size=D_e, num_layers=2, bidirectional=True,
                            dropout=dropout)
        self.lstm_v = nn.LSTM(input_size=hidden_dim, hidden_size=D_e, num_layers=2, bidirectional=True,
                            dropout=dropout)

        self.pos_emb_a = PositionalEncoding(hidden_dim)
        self.pos_emb_v = PositionalEncoding(hidden_dim)
        self.pos_emb_t = PositionalEncoding(hidden_dim)

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net_a = GraphNetwork(2 * D_e, n_classes, n_relations, graph_hidden_size,
                                            dropout, self.no_cuda, self.return_feature)
        self.graph_net_v = GraphNetwork(2 * D_e, n_classes, n_relations, graph_hidden_size,
                                            dropout, self.no_cuda, self.return_feature)
        self.graph_net_l = GraphNetwork(2 * D_e, n_classes, n_relations, graph_hidden_size,
                                            dropout, self.no_cuda, self.return_feature)
        print("construct fourier graph")

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping
        self.gatedatt = MMGatedAttention(2 * D_e + graph_hidden_size, graph_hidden_size, att_type='general')
        self.dropout_ = nn.Dropout(self.dropout)
        if self.att_type == 'concat_subsequently':
            self.smax_fc = nn.Linear(300 * len(self.modals), n_classes)
        elif self.att_type == 'gated':
            if len(self.modals) == 3:
                self.smax_fc = nn.Linear(100 * len(self.modals), n_classes)
            else:
                self.smax_fc = nn.Linear(100, n_classes)
        else:
            self.smax_fc = nn.Linear(2 * D_e + graph_hidden_size, n_classes)

        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * D_e + graph_hidden_size, n_classes)
        )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * D_e + graph_hidden_size, n_classes)
        )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * D_e + graph_hidden_size, n_classes)
        )

    def forward(self, textf, visuf, acouf, qmask, umask, seq_lengths):
        spk_idx = torch.argmax(qmask.permute(1, 0, 2), -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(seq_lengths):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(seq_lengths):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = textf.permute(1, 2, 0).transpose(1, 2)
        acouf = acouf.permute(1, 2, 0).transpose(1, 2)
        visuf = visuf.permute(1, 2, 0).transpose(1, 2)

        textf = self.pos_emb_t(self.linear_t(textf), spk_embeddings)
        acouf = self.pos_emb_t(self.linear_a(acouf), spk_embeddings)
        visuf = self.pos_emb_t(self.linear_v(visuf), spk_embeddings)

        emotions_a, _ = self.lstm_t(textf)
        emotions_v, _ = self.lstm_a(acouf)
        emotions_t, _ = self.lstm_v(visuf)


        features_a, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_a,
                                                                                              qmask,
                                                                                              seq_lengths,
                                                                                              self.window_past,
                                                                                              self.window_future,
                                                                                              self.edge_type_mapping,
                                                                                              self.att_model,
                                                                                              self.no_cuda)

        features_v, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_v,
                                                                                          qmask,
                                                                                          seq_lengths,
                                                                                          self.window_past,
                                                                                          self.window_future,
                                                                                          self.edge_type_mapping,
                                                                                          self.att_model,
                                                                                          self.no_cuda)

        features_l, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_t,
                                                                                          qmask,
                                                                                          seq_lengths,
                                                                                          self.window_past,
                                                                                          self.window_future,
                                                                                          self.edge_type_mapping,
                                                                                          self.att_model,
                                                                                          self.no_cuda)


        emotions_a = self.graph_net_a(features_a, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                      self.nodal_attention, self.avec)


        emotions_v = self.graph_net_v(features_v, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                      self.nodal_attention, self.avec)


        emotions_t = self.graph_net_l(features_l, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                      self.nodal_attention, self.avec)


        t_final_out = self.t_output_layer(emotions_t)
        a_final_out = self.a_output_layer(emotions_a)
        v_final_out = self.v_output_layer(emotions_v)

        t_log_prob = F.log_softmax(t_final_out, 1)
        a_log_prob = F.log_softmax(a_final_out, 1)
        v_log_prob = F.log_softmax(v_final_out, 1)

        kl_t_log_prob = F.log_softmax(t_final_out, 1)
        kl_a_log_prob = F.log_softmax(a_final_out, 1)
        kl_v_log_prob = F.log_softmax(v_final_out, 1)


        emotions_feat = self.gatedatt(emotions_a, emotions_v, emotions_t, self.modals)

        emotions_feat = self.dropout_(emotions_feat)
        final_out = self.smax_fc(emotions_feat)
        all_log_prob = F.log_softmax(final_out, 1)
        all_prob = F.softmax(final_out, 1)
        kl_all_prob = F.softmax(final_out, 1)
        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob, edge_index, edge_norm, edge_type, edge_index_lengths
