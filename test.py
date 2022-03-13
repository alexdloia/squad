"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
import numpy as np
import pickle

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, SCR, SAN
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    if args.model == "scr":
        model = SCR(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    num_candidates=util.NUM_CANDIDATES,
                    only_dcr=args.only_dcr).to(device)
        cand_model = SAN(word_vectors=word_vectors,
                         hidden_size=args.cand_hidden_size,
                         T=args.cand_time_steps).to(device)
    else:
        model = SAN(word_vectors=word_vectors,
                    hidden_size=args.cand_hidden_size,
                    T=args.cand_time_steps).to(device)
    model = nn.DataParallel(model, gpu_ids)
    if args.load_model_path:
        log.info(f'Loading checkpoint from {args.load_model_path}...')
        model, step = util.load_model(model, args.load_model_path, gpu_ids)
    else:
        step = 0

    if args.load_cand_model_path:
        cand_model = nn.DataParallel(cand_model, gpu_ids)
        log.info(f'Loading candidate model checkpoint from {args.load_cand_model_path}...')
        cand_model, cand_step = util.load_model(cand_model, args.load_cand_model_path, gpu_ids)
    else:
        cand_step = 0
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    data_aug_file = vars(args)[f'{args.split}_data_aug_file']
    dataset = SQuAD(record_file, data_aug_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}  # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    k_oracle_file = args.K_pickle
    k_oracles = np.array([1, 2, 3, 5, 10, 20, 50, 100])
    k_oracle_data = np.zeros(k_oracles.shape)
    if args.disposition:
        cand_errors = 0
        dcr_errors = 0
    cnt = 0
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, pos_idxs, ner_idxs, bem_idxs, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            pos_idxs = pos_idxs.to(device)
            ner_idxs = ner_idxs.to(device)
            bem_idxs = bem_idxs.to(device)
            batch_size = cw_idxs.size(0)
            _, p_len = cw_idxs.size()

            # Forward
            y1, y2 = y1.to(device), y2.to(device)
            if args.model == "scr":
                candidates, candidate_scores, _ = util.generate_candidates(cand_model, cw_idxs, qw_idxs, pos_idxs,
                                                                           ner_idxs, bem_idxs,
                                                                           (y1, y2), util.NUM_CANDIDATES,
                                                                           device, train=False)

                logprob_chunks = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs, candidates, candidate_scores)
                c_len = cw_idxs.size()[1]
                c_mask = torch.zeros_like(cw_idxs) != cw_idxs

                log_p1, log_p2 = util.convert_probs(logprob_chunks, candidates, c_len, c_mask, device)
            elif args.disposition:
                candidates, candidate_scores, _ = util.generate_candidates(cand_model, cw_idxs, qw_idxs, pos_idxs,
                                                                           ner_idxs, bem_idxs,
                                                                           (y1, y2), util.NUM_CANDIDATES,
                                                                           device, train=False)

                logprob_chunks = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs, candidates, candidate_scores)
                scr_ans = torch.argmax(logprob_chunks, dim=1)
                c_len = cw_idxs.size()[1]
                c_mask = torch.zeros_like(cw_idxs) != cw_idxs

                for i in range(batch_size):
                    answer_chunk = torch.Tensor([y1[i], y2[i]])

                    found_y = torch.logical_and(candidates[i, :, 0] == answer_chunk[0],
                                                candidates[i, :, 1] == answer_chunk[1]).nonzero()
                    if len(found_y) > 0:
                        # in K-oracle, we are completely correct if one of our candidates is correct
                        idx_correct = found_y[0][0]
                        if scr_ans == idx_correct:
                            print("+", end="")
                        else:
                            print("=", end="")
                    else:
                        print("-", end="")

                log_p1, log_p2 = util.convert_probs(logprob_chunks, candidates, c_len, c_mask, device)
            elif args.K_oracle != 0:
                if args.K_oracle < 0:
                    candidates, _, _ = util.generate_candidates(model, cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs,
                                                                (y1, y2), 100, device, train=False)
                else:
                    candidates, _, _ = util.generate_candidates(model, cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs,
                                                                (y1, y2), args.K_oracle, device, train=False)
                some_log_p1, some_log_p2 = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs)
                log_p1, log_p2 = torch.zeros(batch_size, p_len), torch.zeros(batch_size, p_len)
                log_p1 = log_p1.to(device)
                log_p2 = log_p2.to(device)
                rat = 0.0
                if args.K_oracle < 0:
                    correct = np.zeros(k_oracles.shape)
                for i in range(batch_size):
                    answer_chunk = torch.Tensor([y1[i], y2[i]])

                    found_y = torch.logical_and(candidates[i, :, 0] == answer_chunk[0],
                                                candidates[i, :, 1] == answer_chunk[1]).nonzero()
                    if len(found_y) > 0:
                        # in K-oracle, we are completely correct if one of our candidates is correct
                        idx_correct = found_y[0][0]
                        log_p1[i, candidates[i, idx_correct, 0]] = 1
                        log_p1[i] = torch.log_softmax(log_p1[i], dim=0)
                        log_p2[i, candidates[i, idx_correct, 1]] = 1
                        log_p2[i] = torch.log_softmax(log_p2[i], dim=0)
                        if args.K_oracle < 0:
                            for i, k in enumerate(k_oracles):
                                if idx_correct < k:
                                    correct[i] += 1
                        rat += 1
                        print("+", end="")
                    else:
                        log_p1[i], log_p2[i] = some_log_p1[i], some_log_p2[
                            i]  # otherwise we are just our normal function
                        print("-", end="")
                if args.K_oracle != 0:
                    log.info(rat / batch_size)
                    cnt += batch_size
                    k_oracle_data += correct
            else:
                log_p1, log_p2 = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    if args.K_oracle != 0:
        log.info("Dumping results to", k_oracle_file)
        pickle.dump((k_oracles, k_oracle_data / cnt), open(k_oracle_file, "wb"))
    if args.disposition:
        log.info("Dumping dispositions to", args.disposition)
        pickle.dump((cand_errors, dcr_errors), open(args.disposition, "wb"))

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2, args.q_breakdown,
                                  args.save_dir + '/q_breakdown.json')
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
