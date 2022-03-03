"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class SAN(nn.Module):
    """Stochastic Answer Network (SAN)

    Based on the paper:
    "Stochastic Answer Networks for Machine Reading Comprehension"
    by Xiaodong Liu, Yelong Shen, Kevin Duh and Jianfeng Gao.
    (https://arxiv.org/pdf/1712.03556.pdf).

    Follows a high-level structure as follows:
        - Embedding layer: Embed word indices to get word vectors.
        - Contextual Embedding Layer: Replaced for the puropses of this assignment
        - Memory Generation Layer: construct working memory summary of information from P and Q
        - Answer Module

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size=128, drop_prob=0.4, T=5):
        super(SAN, self).__init__()
        self.hidden_size = hidden_size

        self.encode = layers.LexiconEncoder(word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)

        self.context = layers.ContextualEmbedding(input_size=hidden_size,
                                                  hidden_size=hidden_size,
                                                  num_layers=2,
                                                  drop_prob=drop_prob)

        self.ffn_p = layers.SANFeedForward(input_size=600,
                                           hidden_size=hidden_size,
                                           num_layers=2,
                                           drop_prob=drop_prob)

        self.ffn_q = layers.SANFeedForward(input_size=300,
                                           hidden_size=hidden_size,
                                           num_layers=2,
                                           drop_prob=drop_prob)

        self.memory = layers.MemoryGeneration(hidden_size=hidden_size,
                                              num_layers=1,
                                              drop_prob=drop_prob)

        self.answer = layers.AnswerModule(hidden_size=hidden_size,
                                          drop_prob=drop_prob, T=T)

    def forward(self, pw_idxs, qw_idxs):
        p_mask = torch.zeros_like(pw_idxs) != pw_idxs  # (batch_size, p_len)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs  # (batch_size, q_len)

        p_len, q_len = p_mask.sum(-1), q_mask.sum(-1)

        R_p, R_q = self.encode(pw_idxs, qw_idxs, p_mask, q_mask)  # (batch_size, p_len, 600), (batch_size, q_len, 300)
        print(f"Hidden size: {self.hidden_size}")
        print(f"R_p {R_p.size()} R_q {R_q.size()}")
        E_p = self.ffn_p(R_p)  # (batch_size, p_len, 600) -> (batch_size, p_len, hidden_size) FFN(x) = W_2 ReLU(W_1 x + b_1) + b_2
        E_q = self.ffn_q(R_q)  # (batch_size, q_len, 300) -> (batch_size, q_len, hidden_size) FFN(x) = W_2 ReLU(W_1 x + b_1) + b_2

        print(f"E_p {E_p.size()}, E_q {E_q.size()}")
        H_p = self.context(E_p, p_len)  # (batch_size, p_len, 2 * hidden_size)
        H_q = self.context(E_q, q_len)  # (batch_size, q_len, 2 * hidden_size)

        print(f"H_p {H_p.size()}, H_q {H_q.size()}")
        p_mask_3d = torch.unsqueeze(p_mask, dim=2)
        q_mask_3d = torch.unsqueeze(q_mask, dim=2)
        M = self.memory(H_p, H_q, p_mask_3d, q_mask_3d)  # (batch_size, p_len, 2 * hidden_size)

        print(f"M {M.size()}")
        # at least one step of the answer module MUST be active during training.
        p1, p2 = self.answer(H_p, H_q, M)  # 2 tensors each of shape (batch_size, p_len)

        return p1, p2


class SCR(nn.Module):
    """Smart Chunk Recognition (SCR) model for SQuAD.

    Based on the paper:
    "End-to-End Answer Chunk Extraction and Ranking for Reading Comprehension"
    by Yang Yu, Wei Zhang, Kazi Hasan, Mo Yu, Bing Xiang, Bowen Zhou
    (https://arxiv.org/pdf/1610.09996.pdf).

    Follows a high-level structure as follows:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Candidate Layer: Generate candidate chunks for consideration
        - Chunk Representation Layer: Creates a chunk representation to encode contextual information for each chunk
        - Ranker layer: Rank all chunks using softmax

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        num_candidates: (int): number of candidates to consider
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size, num_candidates, drop_prob=0.):
        super(SCR, self).__init__()
        self.hidden_size = hidden_size
        self.num_candidates = num_candidates
        self.lex = layers.LexiconEncoder(word_vectors=word_vectors,
                                         hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.enc = layers.RNN_GRUEncoder(input_size=hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=1,
                                         drop_prob=drop_prob)

        self.ffn_p = layers.SANFeedForward(input_size=600,
                                           hidden_size=hidden_size,
                                           num_layers=2,
                                           drop_prob=drop_prob)

        self.ffn_q = layers.SANFeedForward(input_size=300,
                                           hidden_size=hidden_size,
                                           num_layers=2,
                                           drop_prob=drop_prob)

        self.att = layers.DCRAttention(hidden_size=hidden_size,
                                       num_layers=1,
                                       drop_prob=drop_prob)

        self.repr = layers.ChunkRepresentationLayer()

        self.rank = layers.RankerLayer()

    def forward(self, cw_idxs, qw_idxs, candidates):
        # candidates is a (batch_size, num_candidates, 2) tensor

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs  # (batch_size, c_len)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs  # (batch_size, q_len())

        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        R_p, R_q = self.lex(cw_idxs, qw_idxs, c_mask, q_mask)  # (batch_size, p_len, 600), (batch_size, q_len, 300)

        c_emb = self.ffn_p(
            R_p)  # (batch_size, p_len, 600) -> (batch_size, p_len, hidden_size) FFN(x) = W_2 ReLU(W_1 x + b_1) + b_2
        q_emb = self.ffn_q(
            R_q)  # (batch_size, q_len, 300) -> (batch_size, q_len, hidden_size) FFN(x) = W_2 ReLU(W_1 x + b_1) + b_2

        hc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        hq = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        gammas = self.att(hc, hq, c_mask, q_mask)  # (batch_size, c_len, 2 * hidden_size)

        # chunk_rep = gamma_bar(m, n) from the paper
        chunk_repr = self.repr(gammas, candidates, hc, hq, c_mask,
                               q_mask)  # (batch_size, num_candidates, 2 * hidden_size)

        out = self.rank(chunk_repr, candidates, hq, q_mask, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        log_p1, log_p2 = out

        return out
