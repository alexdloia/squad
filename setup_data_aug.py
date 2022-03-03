"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

def word_tokenize(sent):
    doc = nlp(sent)
    return [(token.text, token.tag_, token.ent_type_, token.lemma_) for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"][:2]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    example = {"context_tokens": context_tokens,
                               "ques_tokens": ques_tokens,
                               "id": total}
                    examples.append(example)
        print(f"{len(examples)} questions in total")
    return examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_data_aug_features(args, examples, data_type, out_file, tag2idx_dict, ent2idx_dict, is_test=False):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit

    def drop_example(ex, is_test_=False):
        is_test_ = True
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print(f"Converting {data_type} examples to POS/NER/BEM indices...")
    total = 0
    total_ = 0
    meta = {}
    pos_idxs = []
    ner_idxs = []
    bem_idxs = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        context_tokens = example["context_tokens"]
        context_len = len(context_tokens)
        ques_tokens = example["ques_tokens"]

        pos_idx = np.full([para_limit], tag2idx_dict['XX'], dtype=np.int32)
        ner_idx = np.full([para_limit], ent2idx_dict[''], dtype=np.int32)
        bem_idx = np.zeros([para_limit, 3], dtype=np.int32)

        # POS
        pos_idx[:context_len] = np.asarray([tag2idx_dict[tok[1]] for tok in context_tokens])
        pos_idxs.append(pos_idx)

        # NER
        ner_idx[:context_len] = np.asarray([ent2idx_dict[tok[2]] for tok in context_tokens])
        ner_idxs.append(ner_idx)

        # BEM
        for q_tok in ques_tokens:
            stacked = np.asarray(
                [[c_tok[0] == q_tok[0], c_tok[0] == q_tok[0].lower(), c_tok[0] == q_tok[3]] for c_tok in context_tokens],
                dtype=np.int32)  # (context_len, [orig, lower, lemma])
            bem_idx[:context_len] = bem_idx[:context_len] | stacked
        bem_idxs.append(bem_idx)

        ids.append(example["id"])

    np.savez(out_file,
             pos_idxs=pos_idxs,
             ner_idxs=ner_idxs,
             bem_idxs=bem_idxs,
             ids=np.array(ids))
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    train_examples = process_file(args.train_file, "train")
    # POS tagging
    tag_list = nlp.get_pipe("tagger").labels
    tag2idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}
    tag2idx_dict[''] = tag2idx_dict['XX']

    # NER
    ent_list = [name[2:] for name in nlp.entity.move_names[:18]] + ['']
    ent2idx_dict = {ent: idx for idx, ent in enumerate(ent_list)}

    # Process dev and test sets
    dev_examples = process_file(args.dev_file, "dev")
    build_data_aug_features(args, train_examples, "train", args.train_data_aug_file, tag2idx_dict, ent2idx_dict)
    build_data_aug_features(args, dev_examples, "dev", args.dev_data_aug_file, tag2idx_dict, ent2idx_dict)
    if args.include_test_examples:
        test_examples = process_file(args.test_file, "test")
        build_data_aug_features(args, test_examples, "test",
                                            args.test_data_aug_file, tag2idx_dict, ent2idx_dict, is_test=True)


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    # download(args_)

    # Import spacy language model
    nlp = spacy.load("en_core_web_sm")

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args_.glove_dim}d.txt'
    args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args_)
