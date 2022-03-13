"""Utility classes and methods.

Author:
    Chris Chute (chute@stanford.edu)
"""
import logging
import os
import queue
import re
import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import ujson as json
import pickle

import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import Counter

import time

NUM_CANDIDATES = 3
NUM_POS_TAGS = 50 + 1  # adjust for pad value
NUM_NER_TAGS = 19 + 1
POS_UNK = 47
NER_UNK = 18


def plot_question_words(file, name="Model", savepath=None):
    """
        file : the path to the question breakdown json file
        name : name of the model
        savepath : path to save figure too (show if not saved)
    """
    q_breakdown = json.load(open(file, "r"))
    qs = list(q_breakdown.keys())
    plt.plot(qs, [q_breakdown[q]['EM'] for q in qs], label='EM')
    plt.plot(qs, [q_breakdown[q]['F1'] for q in qs], label='F1')
    plt.legend()
    plt.xlabel("Question type")
    plt.ylabel("Evaluation Score")
    plt.title(f"Performance of {name} on Question Types")
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def plot_K(files, names, savepath=None):
    """
        files : the paths to the K_oracle pickle files from test.py
        name : names of the models provided in files
        savepath : path to save figure too (show if not saved)
    """
    for file, name in zip(files, names):
        ks, em_scores = pickle.load(open(file, "rb"))
        plt.plot(ks, em_scores, label=name)
    plt.title("Candidate Model K Oracle Performance")
    plt.xlabel("K")
    plt.ylabel("EM (dev)")
    plt.legend()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def get_lens_from_mask(mask):
    return mask.sum(dim=-1)


def convert_probs(logprob_chunks, candidates, c_len, c_mask, device):
    """
        Converts log probabilities of chunks
        to log_p1, log_p2
    """
    batch_size, num_candidates, _ = candidates.size()
    prob_chunks = logprob_chunks.exp()
    p1 = torch.zeros(batch_size, c_len, device=device)
    p2 = torch.zeros(batch_size, c_len, device=device)
    for i in range(batch_size):
        for j in range(num_candidates):
            p1[i, candidates[i, j, 0]] += prob_chunks[i, j]
            p2[i, candidates[i, j, 1]] += prob_chunks[i, j]

    log_p1 = masked_softmax(p1, c_mask, log_softmax=True)
    log_p2 = masked_softmax(p2, c_mask, log_softmax=True)
    return log_p1, log_p2


def generate_candidates(cand_model, cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs, ys, num_candidates, device,
                        train=True):
    """Given a candidate model, generate the candidates list for input into the SCr model along with a chunk_y which
    represents the solution index.

    Args:
        cand_model (function): Generates a (batch_size x num_candidates x 2) tensor given cq_idxs, qw_idxs
        cw_idxs (tensor): context word indiced
        qw_idxs (tensor): questino word indices
        train (bool, optional): whehter or not we are in train time. The correct chunk is supplied during train time. Defaults to True.

    Returns:
        candidates (tensor): (batch_size x num_candidates x 2) tensor of candidates
    """
    y1, y2 = ys
    batch_size = cw_idxs.size()[0]
    candidates = torch.zeros(batch_size, num_candidates, 2, dtype=torch.long)
    candidate_scores = torch.zeros(batch_size, num_candidates).to(device)
    chunk_y = torch.zeros(batch_size).to(device)
    log_p1, log_p2 = cand_model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs)
    p1, p2 = torch.exp(log_p1), torch.exp(log_p2)
    # loss calc only needed for gradient
    # y1, y2 = y1.to(device), y2.to(device)
    # cand_loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
    # cand_loss_val = cand_loss.item()
    for i in range(batch_size):
        top_candidates, top_candidate_scores = get_candidates_full(p1[i], p2[i], num_candidates)
        candidates[i] = top_candidates
        candidate_scores[i] = top_candidate_scores
        if train:  # only supply correct answer during train time
            answer_chunk = torch.Tensor([y1[i], y2[i]])
            chunky = torch.logical_and(candidates[i, :, 0] == answer_chunk[0],
                                       candidates[i, :, 1] == answer_chunk[1]).nonzero()
            if len(chunky) > 0:
                # the correct answer is simply the index where we found the answer
                chunk_y[i] = chunky[0]
            else:
                candidates[i, -1, :] = answer_chunk
                # the correct answer is where we inserted the answer
                chunk_y[i] = num_candidates - 1

    chunk_y = chunk_y.long()

    return candidates, candidate_scores, chunk_y


def get_candidates_full(p1, p2, num_candidates):
    """Given the probability of start and end chunks,
       this function generates candidates in a smarter way than the basic sample with replacement.

    Args:
        p1 (tensor): (c_len,) tensor of probabilities
        p2 (tensor): (c_len,) tensor of probabilities
    """
    # first, we enumerate all possible chunks as in the DCR paper
    c_len = p1.size()[0]
    num_proposed = num_candidates * 10
    proposed = torch.zeros(num_proposed, 2, dtype=torch.long)
    proposed[:, 0] = torch.tensor(list(torch.utils.data.WeightedRandomSampler(p1, num_proposed, replacement=True)),
                                  dtype=torch.long)
    proposed[:, 1] = torch.tensor(list(torch.utils.data.WeightedRandomSampler(p2, num_proposed, replacement=True)),
                                  dtype=torch.long)
    proposed, _ = torch.sort(proposed, dim=1)
    proposed = torch.unique(proposed, dim=0)

    scores = p1[proposed[:, 0]] * p2[proposed[:, 1]]
    sorted_scores, _ = torch.sort(scores, descending=True)
    cands = proposed[torch.argsort(scores, descending=True)[:num_candidates]]
    scores = sorted_scores[:num_candidates]
    l = len(cands)
    if l < num_candidates:
        r_cands = torch.zeros((num_candidates, 2))
        r_cands[:l] = cands
        r_scores = torch.zeros((num_candidates,))
        r_scores[:l] = scores
        return r_cands, r_scores
    else:
        return cands, scores


def get_candidates_simple(p1, p2, num_candidates):
    candidates = torch.zeros(num_candidates, 2)
    candidates[:, 0] = torch.tensor(list(torch.utils.data.WeightedRandomSampler(p1, num_candidates, replacement=True)),
                                    dtype=torch.long)
    candidates[:, 1] = torch.tensor(list(torch.utils.data.WeightedRandomSampler(p2, num_candidates, replacement=True)),
                                    dtype=torch.long)
    candidates[:, :], _ = torch.sort(candidates, dim=1)

    return candidates


def candidates_enumerate(max_len, p_len):
    candidates = []
    for i in range(p_len):
        for j in range(i, min(i + max_len, p_len)):
            candidates.append([i, j])
    return torch.tensor(candidates)


def chunk_discretize(prob_chunks, candidates):
    """Discretizes prob_chunks in a way to equal the format of the util function discretize

    Args:
        prob_chunks (tensor): (batch_size x num_candidates) tensor of chunk probabilities
        candidates (tensor): (batch_size x num_candidates x 2) tensor of the start/end indices of candidates

    Returns:
        starts: (batch_size,) tensor of the start indicies of the most likely chunks
        ends: (batch_size,) tensor of the end indices of the most likely chunks
    """
    batch_size, _ = prob_chunks.size()
    best_chunks = torch.argmax(prob_chunks, dim=1)
    starts = candidates[range(batch_size), best_chunks, 0]
    ends = candidates[range(batch_size), best_chunks, 1]

    return starts, ends


class SQuAD(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).

    Each item in the dataset is a tuple with the following entries (in order):
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
        - question_idxs: Indices of the words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """

    def __init__(self, data_path, data_aug_path, use_v2=True):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)
        data_aug_set = np.load(data_aug_path)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()
        self.pos_idxs = torch.from_numpy(data_aug_set['pos_idxs']).long()
        self.ner_idxs = torch.from_numpy(data_aug_set['ner_idxs']).long()
        self.bem_idxs = torch.from_numpy(data_aug_set['bem_idxs']).long()

        if use_v2:
            # SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
            batch_size, c_len, w_len = self.context_char_idxs.size()
            ones = torch.ones((batch_size, 1), dtype=torch.int64)
            self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
            self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)
            pos_pad = torch.full((batch_size, 1), POS_UNK)
            self.pos_idxs = torch.cat((pos_pad, self.pos_idxs), dim=1)
            ner_pad = torch.full((batch_size, 1), NER_UNK)
            self.ner_idxs = torch.cat((ner_pad, self.ner_idxs), dim=1)
            zeros = torch.zeros((batch_size, 1, 3), dtype=torch.int64)
            self.bem_idxs = torch.cat((zeros, self.bem_idxs), dim=1)

            ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
            self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
            self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

            self.y1s += 1
            self.y2s += 1

        # SQuAD 1.1: Ignore no-answer examples
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or self.y1s[idx].item() >= 0]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.pos_idxs[idx],
                   self.ner_idxs[idx],
                   self.bem_idxs[idx],
                   self.ids[idx])

        return example

    def __len__(self):
        return len(self.valid_idxs)


def collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `SQuAD.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.

    Args:
        examples (list): List of tuples of the form (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, y1s, y2s, ids).

    Returns:
        examples (tuple): Tuple of tensors (context_idxs, context_char_idxs, question_idxs,
        question_char_idxs, y1s, y2s, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.

    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """

    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    def merge_2d_no_pad(matrices, dtype=torch.int64):
        height, width = matrices[0].size()
        padded = torch.zeros(len(matrices), height, width, dtype=dtype)
        for i, seq in enumerate(matrices):
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # Group by tensor type
    context_idxs, context_char_idxs, \
    question_idxs, question_char_idxs, \
    y1s, y2s, pos_idxs, ner_idxs, bem_idxs, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    _, seq_len = context_idxs.size()
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    pos_idxs = torch.stack(pos_idxs)[:, :seq_len]
    ner_idxs = torch.stack(ner_idxs)[:, :seq_len]
    bem_idxs = torch.stack(bem_idxs)[:, :seq_len, :]
    ids = merge_0d(ids)

    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, pos_idxs, ner_idxs, bem_idxs, ids)


class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def visualize(tbx, pred_dict, eval_path, step, split, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)

    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)

    with open(eval_path, 'r') as eval_file:
        eval_dict = json.load(eval_file)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A'
        example = eval_dict[str(id_)]
        question = example['question']
        context = example['context']
        answers = example['answers']

        gold = answers[0] if answers else 'N/A'
        tbl_fmt = (f'- **Question:** {question}\n'
                   + f'- **Context:** {context}\n'
                   + f'- **Answer:** {gold}\n'
                   + f'- **Prediction:** {pred}')
        tbx.add_text(tag=f'{split}/{i + 1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)


def save_preds(preds, save_dir, file_name='predictions.csv'):
    """Save predictions `preds` to a CSV file named `file_name` in `save_dir`.

    Args:
        preds (list): List of predictions each of the form (id, start, end),
            where id is an example ID, and start/end are indices in the context.
        save_dir (str): Directory in which to save the predictions file.
        file_name (str): File name for the CSV file.

    Returns:
        save_path (str): Path where CSV file was saved.
    """
    # Validate format
    if (not isinstance(preds, list)
            or any(not isinstance(p, tuple) or len(p) != 3 for p in preds)):
        raise ValueError('preds must be a list of tuples (id, start, end)')

    # Make sure predictions are sorted by ID
    preds = sorted(preds, key=lambda p: p[0])

    # Save to a CSV file
    save_path = os.path.join(save_dir, file_name)
    np.savetxt(save_path, np.array(preds), delimiter=',', fmt='%d')

    return save_path


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def discretize(p_start, p_end, max_len=15, no_answer=False):
    """Discretize soft predictions to get start and end indices.

    Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
    subject to `i <= j` and `j - i + 1 <= max_len`.

    Args:
        p_start (torch.Tensor): Soft predictions for start index.
            Shape (batch_size, context_len).
        p_end (torch.Tensor): Soft predictions for end index.
            Shape (batch_size, context_len).
        max_len (int): Maximum length of the discretized prediction.
            I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
        no_answer (bool): Treat 0-index as the no-answer prediction. Consider
            a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
            than the probability assigned to the max-probability span.

    Returns:
        start_idxs (torch.Tensor): Hard predictions for start index.
            Shape (batch_size,)
        end_idxs (torch.Tensor): Hard predictions for end index.
            Shape (batch_size,)
    """
    if p_start.min() < 0 or p_start.max() > 1 \
            or p_end.min() < 0 or p_end.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probabilities
    p_start = p_start.unsqueeze(dim=2)
    p_end = p_end.unsqueeze(dim=1)
    p_joint = torch.matmul(p_start, p_end)  # (batch_size, c_len, c_len)

    # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
    c_len, device = p_start.size(1), p_start.device
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
                                diagonal=max_len)
    if no_answer:
        # Index 0 is no-answer
        p_no_answer = p_joint[:, 0, 0].clone()
        is_legal_pair[0, :] = 0
        is_legal_pair[:, 0] = 0
    else:
        p_no_answer = None
    p_joint *= is_legal_pair

    # Take pair (i, j) that maximizes p_joint
    max_in_row, _ = torch.max(p_joint, dim=2)
    max_in_col, _ = torch.max(p_joint, dim=1)
    start_idxs = torch.argmax(max_in_row, dim=-1)
    end_idxs = torch.argmax(max_in_col, dim=-1)

    if no_answer:
        # Predict no-answer whenever p_no_answer > max_prob
        max_prob, _ = torch.max(max_in_col, dim=-1)
        start_idxs[p_no_answer > max_prob] = 0
        end_idxs[p_no_answer > max_prob] = 0

    return start_idxs, end_idxs


def convert_tokens(eval_dict, qa_id, y_start_list, y_end_list, no_answer):
    """Convert predictions to tokens from the context.

    Args:
        eval_dict (dict): Dictionary with eval info for the dataset. This is
            used to perform the mapping from IDs and indices to actual text.
        qa_id (int): List of QA example IDs.
        y_start_list (list): List of start predictions.
        y_end_list (list): List of end predictions.
        no_answer (bool): Questions can have no answer. E.g., SQuAD 2.0.

    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted answer text.
        sub_dict (dict): Dictionary UUIDs -> predicted answer text (submission).
    """
    pred_dict = {}
    sub_dict = {}
    for qid, y_start, y_end in zip(qa_id, y_start_list, y_end_list):
        context = eval_dict[str(qid)]["context"]
        spans = eval_dict[str(qid)]["spans"]
        uuid = eval_dict[str(qid)]["uuid"]
        if no_answer and (y_start == 0 or y_end == 0):
            pred_dict[str(qid)] = ''
            sub_dict[uuid] = ''
        else:
            if no_answer:
                y_start, y_end = y_start - 1, y_end - 1
            start_idx = spans[y_start][0]
            end_idx = spans[y_end][1]
            pred_dict[str(qid)] = context[start_idx: end_idx]
            sub_dict[uuid] = context[start_idx: end_idx]
    return pred_dict, sub_dict


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction, ''), 0
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append((score, len(ground_truth)))
    return max(scores_for_ground_truths, key=lambda t: t[0])


def eval_dicts(gold_dict, pred_dict, no_answer, q_breakdown_path, a_len_breakdown_path):
    avna = f1 = em = total = 0
    q_breakdown_dict = {"why": {}, "how": {}, "what": {}, "which": {}, "where": {}, "when": {}, "who": {}}
    a_len_breakdown_dict = {str(i): {"EM": [0, 0], "F1": [0, 0]} for i in list(range(11)) + ["11+"]}
    for key, value in pred_dict.items():
        total += 1
        ground_truths = gold_dict[key]['answers']
        prediction = value
        this_em, em_len = metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        this_f1, f1_len = metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
        em += this_em
        f1 += this_f1
        if no_answer:
            this_avna = compute_avna(prediction, ground_truths)
            avna += this_avna

        for q_word, q_eval_dict in q_breakdown_dict.items():
            if q_word in gold_dict[key]['question'].lower():
                q_eval_dict["total"] = q_eval_dict.get("total", 0) + 1
                q_eval_dict["EM"] = q_eval_dict.get("EM", 0) + this_em
                q_eval_dict["F1"] = q_eval_dict.get("F1", 0) + this_f1
                if no_answer:
                    q_eval_dict["AvNA"] = q_eval_dict.get("AvNA", 0) + this_avna

        # a len
        em_key = str(em_len) if em_len <= 10 else "11+"
        a_len_breakdown_dict[em_key]["EM"][0] += this_em
        a_len_breakdown_dict[em_key]["EM"][1] += 1
        f1_key = str(f1_len) if f1_len <= 10 else "11+"
        a_len_breakdown_dict[f1_key]["F1"][0] += this_f1
        a_len_breakdown_dict[f1_key]["F1"][1] += 1

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}

    if no_answer:
        eval_dict['AvNA'] = 100. * avna / total

    for q_word, q_eval_dict in q_breakdown_dict.items():
        q_eval_dict["EM"] = 100. * q_eval_dict["EM"] / q_eval_dict["total"]
        q_eval_dict["F1"] = 100. * q_eval_dict["F1"] / q_eval_dict["total"]
        q_eval_dict["AvNA"] = 100. * q_eval_dict["AvNA"] / q_eval_dict["total"]
    json.dump(q_breakdown_dict, open(q_breakdown_path, 'w'))

    for a_len, a_len_eval_dict in a_len_breakdown_dict.items():
        a_len_eval_dict["EM"][0] = 100. * a_len_eval_dict["EM"][0] / max(1, a_len_eval_dict["EM"][1])
        a_len_eval_dict["F1"][0] = 100. * a_len_eval_dict["F1"][0] / max(1, a_len_eval_dict["F1"][1])
    json.dump(a_len_breakdown_dict, open(a_len_breakdown_path, 'w'))

    return eval_dict


def compute_avna(prediction, ground_truths):
    """Compute answer vs. no-answer accuracy."""
    return float(bool(prediction) == bool(ground_truths))


# All methods below this line are from the official SQuAD 2.0 eval script
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def test():
    test = "cand_full"
    if test == "cand_full":
        p1 = torch.tensor([0.0, 0.1, 0.1, 0.2, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0])
        p2 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.3, 0.1])
        num_candidates = 20
        top_candidates, _ = get_candidates_full(p1, p2, num_candidates)
        print(top_candidates)


if __name__ == "__main__":
    test()
