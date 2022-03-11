"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import SCR, SAN
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD, NUM_CANDIDATES


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    print(word_vectors[0:100])

    # Get model
    log.info('Building model...')
    log.info(f"Running with model {args.model}")
    cand_model = None
    if args.model == "scr":
        model = SCR(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    num_candidates=NUM_CANDIDATES,
                    drop_prob=args.drop_prob,
                    only_dcr=args.only_dcr).to(device)
        cand_model = SAN(word_vectors=word_vectors,
                         hidden_size=args.cand_hidden_size,
                         drop_prob=args.drop_prob, T=args.cand_time_steps).to(device)
    else:
        model = SAN(word_vectors=word_vectors,
                    hidden_size=args.cand_hidden_size,
                    drop_prob=args.drop_prob,
                    T=args.cand_time_steps).to(device)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_model_path:
        log.info(f'Loading checkpoint from {args.load_model_path}...')
        model, step = util.load_model(model, args.load_model_path, args.gpu_ids)
    else:
        step = 0

    if args.load_cand_model_path:
        cand_model = nn.DataParallel(cand_model, args.gpu_ids)
        log.info(f'Loading candidate model checkpoint from {args.load_cand_model_path}...')
        cand_model, cand_step = util.load_model(cand_model, args.load_cand_model_path, args.gpu_ids)
    else:
        cand_step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr, weight_decay=args.l2_wd)
    if args.lr_sched == 'san':
        scheduler = sched.LambdaLR(optimizer, lambda epochs: 0.5 ** (epochs // 10))  # SAN LR
    else:
        scheduler = sched.LambdaLR(optimizer, lambda epochs: max(0.5 ** 5, 0.5 ** (epochs // 5)))  # custom LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.train_data_aug_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.dev_data_aug_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # torch.autograd.set_detect_anomaly(True)
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        scheduler.step()
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, pos_idxs, ner_idxs, bem_idxs, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                pos_idxs = pos_idxs.to(device)
                ner_idxs = ner_idxs.to(device)
                bem_idxs = bem_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                if args.model == "scr":
                    candidates, candidate_scores, chunk_y = util.generate_candidates(cand_model, cw_idxs, qw_idxs,
                                                                                     pos_idxs, ner_idxs,
                                                                                     bem_idxs, (y1, y2),
                                                                                     NUM_CANDIDATES, device, train=True)
                    chunk_y.to(device)
                    candidate_scores.to(device)
                    logprob_chunks = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs, candidates, candidate_scores)

                    logprob_chunks.to(device)
                    # weighted_logprobs = torch.mul(logprob_chunks, candidate_scores)
                    # weighted_logprobs.to(device)
                    loss = F.nll_loss(logprob_chunks, chunk_y)
                    loss_val = loss.item()

                else:
                    # Forward
                    log_p1, log_p2 = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs)
                    y1, y2 = y1.to(device), y2.to(device)
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                    loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    if args.model == 'scr':
                        results, pred_dict = evaluate(model, dev_loader, device,
                                                      args.dev_eval_file,
                                                      args.max_ans_len,
                                                      args.use_squad_v2,
                                                      cand_model,
                                                      args.model == "scr")
                        # Checking alpha
                        for child in model.children():
                            for name, param in child.named_parameters():
                                if name == "rank.alpha":
                                    log.info(name)
                                    log.info(param)
                    else:
                        results, pred_dict = evaluate(model, dev_loader, device,
                                                      args.dev_eval_file,
                                                      args.max_ans_len,
                                                      args.use_squad_v2,
                                                      None,
                                                      args.model == "scr")
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2, cand_model=None, chunk=False):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    # Checking alpha
    if chunk:
        for child in model.children():
            for name, param in child.named_parameters():
                if name == "rank.alpha":
                    print(name)
                    print(param)

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, pos_idxs, ner_idxs, bem_idxs, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            pos_idxs = pos_idxs.to(device)
            ner_idxs = ner_idxs.to(device)
            bem_idxs = bem_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            if chunk:
                candidates, candidate_scores, _ = util.generate_candidates(cand_model, cw_idxs, qw_idxs, pos_idxs,
                                                                           ner_idxs,
                                                                           bem_idxs, (y1, y2), NUM_CANDIDATES,
                                                                           device, train=False)
                logprob_chunks = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs, candidates, candidate_scores)
                candidate_scores.to(device)
                c_len = cw_idxs.size()[1]
                c_mask = torch.zeros_like(cw_idxs) != cw_idxs
                log_p1, log_p2 = util.convert_probs(logprob_chunks, candidates, c_len, c_mask, device)

            else:
                log_p1, log_p2 = model(cw_idxs, qw_idxs, pos_idxs, ner_idxs, bem_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
