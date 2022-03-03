"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

from multiprocessing.sharedctypes import Value
from random import randrange
from turtle import backward
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import masked_softmax, tag_list, ent_list, \
    get_binary_exact_match_features, indices_to_pos_ner_one_hots, get_lens_from_mask
from torch.distributions.bernoulli import Bernoulli


class POSNERTagging(nn.Module):
    def __init__(self, pos_emb_size=9, ner_emb_size=8):
        super(POSNERTagging, self).__init__()
        self.pos_map = nn.Linear(len(tag_list), pos_emb_size)
        self.ner_map = nn.Linear(len(ent_list), ner_emb_size)

    def forward(self, idxs, mask):
        """

        Args:
            idxs: (batch_size, seq_len)
            mask: (batch_size, seq_len)

        Returns: POS embedding

        """
        pos_one_hots, ner_one_hots = indices_to_pos_ner_one_hots(idxs, mask)
        return self.pos_map(pos_one_hots.to(torch.float32)), self.ner_map(ner_one_hots.to(torch.float32))


class LexiconEncoder(nn.Module):
    def __init__(self, hidden_size, drop_prob, word_vectors):
        super(LexiconEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.posnertagging = POSNERTagging()
        self.w0 = nn.Linear(300, 280, bias=False)
        self.g_func = nn.Sequential(nn.Linear(300, 280, bias=False), nn.ReLU())

    def forward(self, pw_idxs, qw_idxs, p_mask, q_mask):
        # step 1: embed x (batch_size, p_len)
        embed = self.embed(pw_idxs)  # (batch_size, p_len, embed_size)

        # step 2 & 3: get POS and NER tagging for x
        pos, ner = self.posnertagging(pw_idxs, p_mask)  # (batch_size, p_len, 9), (batch_size, p_len, 8)

        # step 4: get binary exact match feature
        # this feature is 3 dimensions for 3 kinds of matching between the pw_idxs and the qw_idxs
        bem = get_binary_exact_match_features(pw_idxs, qw_idxs, p_mask, q_mask)  # (batch_size, p_len, 3)
        # remember that p_mask is a mask over what words are actually there!!

        # step 5: get question-enhanced word embedding. requires some math
        # Define f_align(p_i) = sum(gamma[i, j] * g(GLOVE(q_j)) for j in range(qi_len)
        # g(.) is a 280-dimensional single layer g(x) = ReLU(W_0 x)
        # gamma[i, j] = (g(GLOVE(p_i)) * g(GLOVE(q_j))).exp() / sum(np.exp(g(GLOVE(p_i)) * g(GLOVE(q_j))) for j in range(qi_len))
        # remember that the length of q is variable based on q_mask
        R_q = self.embed(qw_idxs)  # (batch_size, q_len, embed_size)
        # gammas (batch_size, p_len, q_len)
        g_p = self.g_func(embed)  # (batch_size, p_len, 280)
        g_q = self.g_func(R_q)  # (batch_size, q_len, 280)
        pregammas = torch.bmm(g_p, g_q.transpose(1, 2))  # (batch_size, p_len, q_len)
        gammas = F.softmax(pregammas, dim=-1)
        # align = f_align(p_i) for i in p_len for b in batch_size (batch_size, p_len, 280)
        align = torch.bmm(gammas, g_q)  # (batch_size, p_len, 280)

        R_p = torch.cat((embed, align, pos, ner, bem), dim=-1)  # (batch_size, p_len, embed_size + 300)

        return R_p, R_q


class ContextualEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_prob):
        super(ContextualEmbedding, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.enc = RNNEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            drop_prob=drop_prob
        )

    def forward(self, x, lengths):
        # given x (batch_size, hidden_size, seq_len)

        # the paper used a pretrained BiLSTM, guess we need to replace this with something
        # for now, we can just train our own BiLSTM?
        # see BiDAF / SCR for an example

        return self.enc(x, lengths)


class SANFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_prob):
        super(SANFeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.W_1 = nn.Linear(input_size, hidden_size)

        if num_layers == 2:
            self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.W_1(x)
        x = self.relu(x)
        if self.num_layers == 2:
            x = self.W_2(x)
        return x


class DotProductAttention(nn.Module):
    """
    Dot Product Attention modeled off of
    -https://www.tandfonline.com/doi/full/10.1080/00051144.2020.1809221
    """

    def __init__(self, hidden_size, drop_prob):
        super(DotProductAttention, self).__init__()
        self.Qkey = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.P = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, H_qhat, H_phat, q_mask):
        P = self.relu(self.P(H_phat))  # shape (batch_size, p_len, hidden_size)
        Qkey = self.relu(self.Qkey(H_qhat))  # shape (batch_size, q_len, hidden_size)
        pt = torch.transpose(P, 1, 2) # shape (batch_size, hidden_size, p_len)
        a = torch.matmul(Qkey, pt) # shape (batch_size, q_len, p_len)
        C = masked_softmax(a, dim=1, mask=q_mask)  # shape (batch_size, q_len, p_len)
        return C


class MemoryGeneration(nn.Module):
    def __init__(self, hidden_size, num_layers, drop_prob):
        super(MemoryGeneration, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.ffn_q = SANFeedForward(2 * hidden_size, hidden_size, 1, drop_prob)
        self.ffn_p = SANFeedForward(2 * hidden_size, hidden_size, 1, drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.f_attn = DotProductAttention(hidden_size, drop_prob)
        self.lstm = nn.LSTM(8 * hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True,
                            dropout=drop_prob if num_layers > 1 else 0.)
        self.Up1 = nn.Linear(in_features=4 * hidden_size, out_features=4 * hidden_size, bias=True)
        self.Up2 = nn.Linear(in_features=4 * hidden_size, out_features=4 * hidden_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, H_p, H_q, p_mask, q_mask):
        H_qhat = self.ffn_q(H_q)  # (batch_size, q_len, hidden_size) 
        H_phat = self.ffn_p(H_p)  # (batch_size, p_len, hidden_size)
        # _, p_len, _ = H_p.size()

        C = self.dropout(self.f_attn(H_qhat, H_phat, q_mask))  # (batch_size, q_len, p_len)
        # attention from https://arxiv.org/pdf/1706.03762.pdf

        Ct = torch.transpose(C, 1, 2) # (batch_size, p_len, q_len)
        dp_att = torch.matmul(Ct, H_q)  # (batch_size, p_len, 2*hidden_size)
        U_p = torch.cat((H_p, dp_att), dim=-1)  # (batch_size, p_len, 4*hidden_size)

        U_p1 = self.Up1(U_p)  # (batch_size, p_len, 4*hidden_size)
        U_p2 = self.Up2(U_p)  # (batch_size, p_len, 4*hidden_size)
        U_p2 = torch.transpose(U_p2, 1, 2)

        U_phat = U_p1 @ U_p2  # (batch_size, p_len, p_len)

        U_phat = masked_softmax(U_phat, dim=1, mask=p_mask)  # (batch_size, p_len, p_len)

        U_phat = U_phat * (1 - torch.eye(p_len).to(U_phat.device))

        U_phat = torch.matmul(self.dropout(U_phat), U_p)  # (batch_size, p_len, hidden_size)

        U = torch.cat((U_p, U_phat), dim=-1)  # (batch_size, p_len, 8*hidden_size)
        M, _ = self.lstm(U)

        return M

class AnswerModule(nn.Module):
    def __init__(self, hidden_size, drop_prob, T):
        super(AnswerModule, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.T = T
        self.W_4 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.W_5 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.W_6 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.W_7 = nn.Linear(2 * hidden_size, 4 * hidden_size, bias=False)
        self.gru = nn.GRU(2 * hidden_size, 2 * hidden_size, num_layers=1,
                          batch_first=True,
                          bidirectional=False,
                          dropout=0.)

    def forward(self, H_p, H_q, M):
        # answer module computes over T memory steps and outputs answer span
        batch_size, q_len, d_2 = H_q.size()
        _, p_len, _ = H_p.size()

        s = [H_p.new_zeros(size=(batch_size, 2 * self.hidden_size)) for _ in range(self.T)]

        alpha = self.W_4(H_q)  # (batch_size, q_len, 1)
        alpha = torch.squeeze(alpha, dim=2)  # (batch_size, q_len)
        alpha = torch.softmax(alpha, dim=1)
        alpha = torch.unsqueeze(alpha, dim=1)  # (batch_size, q_len, 1) for bmm
        alpha = torch.bmm(alpha, H_q)
        s[0] = torch.squeeze(alpha, dim=1)  # sum_j \alpha_j (H_q)_j for each batch

        for t in range(1, self.T):
            s_tmp = torch.unsqueeze(s[t - 1], dim=2)  # (batch_size, 2 * hidden_size, 1) for bmm
            beta = torch.bmm(self.W_5(M), s_tmp)
            beta = torch.squeeze(beta, dim=2)  # (batch_size, p_len)
            beta = torch.softmax(beta, dim=1)  # softmax across the non-batch dimension. (batch_size, p_len)

            beta = torch.unsqueeze(beta, dim=1)
            x = torch.bmm(beta, M)  # sum beta_j M_j for all j, all batch (batch_size, 2 * hidden_size)
            x = torch.squeeze(x, dim=1)  # (batch_size, 2 * hidden_size)
            s_tmp = torch.unsqueeze(s[t - 1], dim=1)  # (batch_size, 1, 2 * hidden_size)
            x = torch.unsqueeze(x, dim=0)  # (1, batch_size, 2 * hidden_size)
            s_tmp, _ = self.gru(s_tmp, x)
            s[t] = torch.squeeze(s_tmp, dim=1)  # (batch_size, 2 * hidden_size)

        # Finally, we get our probability distributions

        if self.training:  # dropout during training
            chosen_t = torch.zeros(p_len)
            bernoulli = Bernoulli(torch.tensor([self.drop_prob] * self.T))
            while sum(chosen_t) == 0:  # while no time step are chosen, rechoose
                chosen_t = bernoulli.sample()
        else:
            chosen_t = torch.ones(p_len)

        p1 = H_p.new_zeros(size=(batch_size, p_len))
        p2 = H_p.new_zeros(size=(batch_size, p_len))
        for t in range(self.T):
            if not chosen_t[t]:
                continue

            s_tmp = torch.unsqueeze(s[t], dim=2)  # (batch_size, 2 * hidden_size, 1)
            w6_tmp = torch.bmm(self.W_6(M), s_tmp)
            w6_tmp = torch.squeeze(w6_tmp, dim=2)
            p1_tmp = torch.softmax(w6_tmp, dim=1)
            p1_tmp_3d = torch.unsqueeze(p1_tmp, dim=1)  # (batch_size, 1, p_len)

            s2 = torch.bmm(p1_tmp_3d, M)  # (batch_size, 1, 2 * hidden_size)
            s2 = torch.squeeze(s2, dim=1)
            s2 = torch.cat((s[t], s2), dim=1)
            s2 = torch.unsqueeze(s2, dim=2)  # (batch_size, 4 * hidden_size)

            w7_tmp = torch.bmm(self.W_7(M), s2)  # (batch_size, p_len, 1)
            w7_tmp = torch.squeeze(w7_tmp, dim=2)
            p2_tmp = torch.softmax(w7_tmp, dim=1)  # (batch_size, p_len)

            p1 += p1_tmp
            p2 += p2_tmp

        p1 /= sum(chosen_t)  # normalize our probabilities by how many distributions we summed
        p2 /= sum(chosen_t)

        return p1.log(), p2.log()  # return as log probabilities for their code scaffolding


class CustomEmbedding(nn.Module):
    """Embedding layer used by DCR

    Concatenates the embedding of the word with
    some meta-information about the word.
    POS, NER, IsInQuestion, IsLemmaInQuestion, IsCapitalized
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(CustomEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        """
            Embed the word as in Embedding,
            but also append some other information about the word to the vector.

            x is a (batch_size, seq_len, embed_size) Tensor
            note, len here could be the length of the paragraph of the question
        """
        emb = self.embed(x)  # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)
        # TODO add meta-information, remove highway embedding?

        return emb


class RNN_GRUEncoder(nn.Module):
    """
        bidirectional RNN layer with gated recurrent units (GRU).
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.):
        super(RNN_GRUEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        """
            Let x_t be the t-th word vector.
            r_t = \sigma(W_r x_t + U_r h_{t-1}) # attention over hidden state
            u_t = \sigma(W_u x_t + U_u h_{t-1}) # how much to forget / weight of new hidden state
            h'_t = tanh(W x_t + U (r_t .* h_{t-1})) # next state, .* = element-wise multiplication
            h_t = (1 - u_t) h_{t-1} + u_t h'_t # next state is an interpolation of last one and h', weight is u_t

            h_t, r_t, u_t \in R^d (d = hidden state dimension)
            W_r, W_u, W \in R^(n x d) and U_r, U_u, U \in R^(d x d) are the model parameters

            After doing both directions of the RNN:
            h_t = [h_t (forward) ; h_t (backward)] # concatenate the forward and backward directions at this point
        """
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)  # shape (batch_size, seq_len, 2 * hidden_size)

        return x


class DCRAttention(nn.Module):
    """Attention originally used by DCR.
    """

    def __init__(self, hidden_size, num_layers=1, drop_prob=0.):
        super(DCRAttention, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.GRU(4 * hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, hp, hq, c_mask, q_mask):
        """
            Attention on each p_j (jth word of the context p_i)
            \alpha_jk = hp_j * hq_k
            \beta_j = sum over k in Q of \alpha_jk hq_k
            v_j = [hp_j ; \beta_j]

            v_j \in R^4d

            We then apply a second RNN_GRUEncoder to the v_js
            to get \gamma_j (each should be each to [\gamma_j (forward) ; \gamma_j (backward)])
            for each word.

        """
        batch_size, p_len, _ = hp.size()
        _, q_len, _ = hq.size()

        alpha = torch.bmm(hp, torch.transpose(hq, 1,
                                              2))  # (batch_size x p_len x 2d) x (batch_size x 2d x q_len) = (batch_size x p_len x q_len)

        beta = torch.bmm(alpha,
                         hq)  # batch_size x p_len x q_len) times (batch_size x q_len x 2d) = (batch_size x p_len x 2d)

        V = torch.cat((hp, beta),
                      dim=2)  # (batch_size x p_len x 2d) concat (batch_size x p_len x 2d) = (batch_size, p_len, 4d)

        gammas, _ = self.rnn(V)  # output from rnn is (batch_size x p_len x 2d)

        return gammas


class CandidateLayer(nn.Module):
    """ new, custom layer that chooses a selection of potential chunk candidates
    """

    def __init__(self, num_candidates):
        super(CandidateLayer, self).__init__()
        self.num_candidates = num_candidates

    def forward(self, c, q, c_mask, q_mask, num_candidates):
        """
            Still need to figure out what this layer looks like.
        """

        # return candidates ( a list of tuples of indices i.e. [(3, 5), (3, 6), (4, 6), (11, 16)])
        # each m, n pair should have m < n and signify the range m to n inclusive.
        # might want to format this as a (num_candidates x 2) tensor
        batch_size = c.size()[0]
        r = torch.ones(batch_size, num_candidates, 2, dtype=torch.long)
        r[:, :, 0] = 0
        r[:, :, 1] = 1
        return r  # make sure return value is a Long Tensor!
        # TODO


class ChunkRepresentationLayer(nn.Module):
    """ Create a chunk representation for each candidate chunk
    """

    def __init__(self):
        super(ChunkRepresentationLayer, self).__init__()

    def forward(self, gammas: torch.Tensor, candidates: torch.Tensor, p_enc, q_enc, p_mask, q_mask):
        """
            For each candidate chunk c(m, n):
            we construct \gamma(m, n) = [\gamma_m (forward) ; \gamma_n (backward)]

            gammas (batch_size x p_len x 2d) tensor
            candidates (batch_size x num_candidates x 2) tensor

            candidates is an index into gammas for us to use.

            return chunk_repr, a (batch_size x num_candidates x 2d) tensor)

        """
        batch_size, p_len, t = gammas.size()
        _, num_candidates, _ = candidates.size()
        d = t // 2
        forward_gammas = gammas[:, :, :d]
        backward_gammas = gammas[:, :, d:]
        first_word_idx = candidates[:, :, 0].unsqueeze(-1)  # batch_size x num_candidates x 1
        last_word_idx = candidates[:, :, 1].unsqueeze(-1)  # batch_size x num_candidates x 1

        first_forward_gammas = torch.gather(forward_gammas, 1,
                                            first_word_idx.expand(-1, -1,
                                                                  d))  # batch_size x num_candidates x d
        last_backward_gammas = torch.gather(backward_gammas, 1,
                                            last_word_idx.expand(-1, -1,
                                                                 d))  # batch_size x num_candidates x d
        chunk_repr = torch.cat((first_forward_gammas, last_backward_gammas), dim=-1)  # batch_size x num_candidates x 2d

        return chunk_repr  # (batch_size x num_candidates x 2d) tensor


class RankerLayer(nn.Module):
    """ Rank the candidate chunks with softmax
    """

    def __init__(self):
        super(RankerLayer, self).__init__()

    def forward(self, chunk_repr: torch.Tensor, candidates: torch.Tensor, hq: torch.Tensor, q_mask: torch.Tensor,
                c_mask: torch.Tensor):
        """
            chunk_repr (batch_size x num_candidates x 2d) tensor
            candidates (batch_size x num_candidates x 2) tensor
            hq (batch_size, q_len, 2d) tensor
            q_mask : mask on q, (batch_size x q_len) mask - I think?

            Let b = |Q|
            P[c(m , n)] = softmax(\gamma(m, n) * [hq_b (forward) ; hq_1 (backward)])

            When training, we try to minimize:

            L = - \sum (training examples) log (A | P, Q)
            Where A is the correct answer chunk.

            ONLY train on examples where the correct answer is a candidate chunk!!
        """
        # batch_size, c_len = c_mask.size()
        # p1 = torch.ones(batch_size, c_len)
        # p2 = torch.ones(batch_size, c_len)
        # F.normalize(p1)
        # F.normalize(p2)
        #
        # return torch.log(p1), torch.log(p2)

        batch_size, q_len, t = hq.size()
        d = t // 2
        q_last_indices = get_lens_from_mask(q_mask) - 1  # (batch_size,)
        last_forwards_hq = hq[torch.arange(batch_size), q_last_indices, :d]  # batch_size x d
        first_backwards_hq = hq[:, 0, d:].squeeze(1)
        H = torch.cat((last_forwards_hq, first_backwards_hq), dim=-1).unsqueeze(-1)  # batch_size x 2d x 1
        cos_sim = torch.bmm(chunk_repr,
                            H)  # (batch_size x num_candidates x 2d) x (batch_size x 2d x 1) = (batch_size x num_candidates x 1)
        cos_sim = cos_sim.squeeze(-1)
        sm = F.log_softmax(cos_sim, dim=-1)
        return sm


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)  # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


if __name__ == "__main__":
    test = "MemoryGeneration"
    batch_size, num_candidates, d, p_len, q_len, T, drop_prob = 5, 4, 3, 10, 15, 5, 0.4
    if test == "RankerLayer":
        """
                    Ranker Layer:
                    chunk_repr (batch_size x num_candidates x 2d) tensor
                    candidates (batch_size x num_candidates x 2) tensor
                    hq (batch_size, q_len, 2d) tensor
                    q_mask : mask on q, (batch_size x q_len) mask - I think?

                    Let b = |Q|
                    P[c(m , n)] = softmax(\gamma(m, n) * [hq_b (forward) ; hq_1 (backward)])

                    When training, we try to minimize:

                    L = - \sum (training examples) log (A | P, Q)
                    Where A is the correct answer chunk.

                    ONLY train on examples where the correct answer is a candidate chunk!!
                """
        rank = RankerLayer()
        chunk_repr = torch.randn(batch_size, num_candidates, 2 * d)
        candidates = torch.randn(batch_size, num_candidates, 2)
        hq = torch.randn(batch_size, q_len, 2 * d)
        q_mask = torch.ones(batch_size, q_len)
        q_mask[:, 5:] = 0
        q_mask = torch.zeros_like(q_mask) != q_mask
        print(rank(chunk_repr, candidates, hq, q_mask, None))
    elif test == "ChunkRepresentationLayer":
        """
                    For each candidate chunk c(m, n):
                    we construct \gamma(m, n) = [\gamma_m (forward) ; \gamma_n (backward)]

                    gammas (batch_size x p_len x 2d) tensor
                    candidates (batch_size x num_candidates x 2) tensor

                    candidates is an index into gammas for us to use.

                    return chunk_repr, a (batch_size x num_candidates x 2d) tensor)

                """
        crepr = ChunkRepresentationLayer()
        gammas = torch.randn(batch_size, p_len, 2 * d)
        candidates = torch.randint(p_len, size=(batch_size, num_candidates, 2))
        print(crepr(gammas, candidates, None, None, None, None))
    elif test == "DCRAttention":
        datt = DCRAttention(d)
        hp = torch.randn(batch_size, p_len, 2 * d)
        hq = torch.randn(batch_size, q_len, 2 * d)
        print(datt(hp, hq, None, None))
    elif test == "AnswerModule":
        d = 128
        answer = AnswerModule(d, drop_prob, T)
        H_p = torch.randn(batch_size, p_len, 2 * d)
        H_q = torch.randn(batch_size, q_len, 2 * d)
        M = torch.rand(batch_size, p_len, 2 * d)

        log_p1, log_p2 = answer(H_p, H_q, M)
        print(log_p1, log_p2)
    elif test == "MemoryGeneration":
        memGen = MemoryGeneration(hidden_size=128, num_layers=1, drop_prob=.4)
        q_len = 10
        p_len = 30
        H_q = torch.randn(batch_size, q_len, 2 * 128)
        H_p = torch.randn(batch_size, p_len, 2 * 128)
        p_mask=torch.ones(batch_size, p_len, 1)
        q_mask=torch.ones(batch_size, q_len, 1)
        print(memGen(H_p, H_q, p_mask=p_mask, q_mask=q_mask))
    elif test == "ContextualEmbedding":
        context = ContextualEmbedding(input_size=d, hidden_size=d, drop_prob=drop_prob, num_layers=2)
        x = torch.randn(batch_size, p_len, d)
        lengths = torch.randint(1, p_len + 1, (batch_size,))
        print(context(x, lengths))
