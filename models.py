"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class SCR(nn.Module):
    """Smart Chunk Recognition (SCR) model for SQuAD.

    Based on the paper:
    "End-to-End Answer Chunk Extraction and Ranking for Reading Comprehension"
    by Yang Yu, Wei Zhang, Kazi Hasan, Mo Yu, Bing Xiang, Bowen Zhou
    (https://arxiv.org/pdf/1610.09996.pdf).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Candidate Layer: Generate candidate chunks for consideration
        - Chunk Representation Layer: Creates a chunk representation to encode contextual information for each chunk
        - Ranker layer: Rank all chunks using softmax

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, num_canidates, drop_prob=0.):
        super(SCR, self).__init__()
        self.emb = layers.CustomEmbedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNN_GRUEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.DCRAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.cand = layers.CandidateLayer(num_canidates=num_canidates)

        self.repr = layers.ChunkRepresentationLayer()

        self.out = layers.RankerLayer()

    def forward(self, pw_idxs, qw_idxs):
        p_mask = torch.zeros_like(pw_idxs) != pw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        p_len, q_len = p_mask.sum(-1), q_mask.sum(-1)

        p_emb = self.emb(pw_idxs)         # (batch_size, p_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        p_enc = self.enc(p_emb, p_len)    # (batch_size, p_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(p_enc, q_enc, p_mask, q_mask)    # (batch_size, p_len, 4 * hidden_size)

        candidates = self.cand(pw_idxs, qw_idxs) # (batch_size, num_candidates, 2)

        # ONLY train on examples where the correct answer is a candidate chunk!!
        # maybe check that somehow?

        gammas = self.repr(att, candidates, p_enc, q_enc) # (batch_size, num_candidates, 2 * hidden_size)

        out = self.out(gammas, q_enc)  # 2 tensors, each (batch_size, p_len)

        log_p1, log_p2 = out # for debugging htere

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
        print(cw_idxs.size(), qw_idxs.size())
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        print(c_emb.size(), q_emb.size())

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        log_p1, log_p2 = out
        print(log_p1.size(), log_p2.size())
        raise ValueError("Yeah we're done here")

        return out
