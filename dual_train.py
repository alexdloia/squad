"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import chunk
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
from models import BiDAF, SCR
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

NUM_CANDIDATES = 20

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

    # Get model
    log.info('Building model...')
    print(f"Running with model {args.model}")
    if args.model == "scr":
        model = SCR(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    num_candidates=NUM_CANDIDATES,
                    drop_prob=args.drop_prob)
        cand_model = BiDAF(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    drop_prob=args.drop_prob)
    else:
        model = BiDAF(word_vectors=word_vectors,
                    hidden_size=args.hidden_size,
                    drop_prob=args.drop_prob)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.model == "scr":
        cand_model = nn.DataParallel(cand_model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    if args.model == "scr":
        cand_model = cand_model.to(device)
        cand_model.train()
        cand_ema = util.EMA(cand_model, args.ema_decay)
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    if args.model == 'scr':
        cand_optimizer = optim.Adadelta(cand_model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
        cand_scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()


                cand_optimizer.zero_grad()

                candidates = torch.zeros(batch_size, NUM_CANDIDATES, 2, dtype=torch.long)
                chunk_y = torch.zeros(batch_size)
                log_p1, log_p2 = cand_model(cw_idxs, qw_idxs)
                p1, p2 = torch.exp(log_p1), torch.exp(log_p2)
                y1, y2 = y1.to(device), y2.to(device)
                cand_loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                cand_loss_val = cand_loss.item()
                for i in range(args.batch_size):
                        # (batch_size, c_len) -> (c_len,)

                    # for now, random sampling WITH replacement for candidate generation
                    candidates[i, :, 0] = torch.tensor(list(torch.utils.data.WeightedRandomSampler(p1[i], NUM_CANDIDATES, replacement=True)), dtype=torch.long)
                    candidates[i, :, 1] = torch.tensor(list(torch.utils.data.WeightedRandomSampler(p2[i], NUM_CANDIDATES, replacement=True)), dtype=torch.long)
                    candidates[i, :, :], _ = torch.sort(candidates[i, :, :], axis=1)

                    answer_chunk = torch.Tensor([y1[i], y2[i]])
                    chunky = torch.logical_and(candidates[i, :, 0] == answer_chunk[0], candidates[i, :, 1] == answer_chunk[1]).nonzero()
                    if len(chunky) > 0:
                        # the correct answer is simply the index where we found the answer
                        chunk_y[i] = chunky[0]
                    else:
                        candidates[i, -1, :] = answer_chunk
                        # the correct answer is where we inserted the answer
                        chunk_y[i] = NUM_CANDIDATES - 1
                    print(p1)
                    print(p2)
                    print(candidates[i, :, :])
                    print(answer_chunk)
                    print(chunk_y)
                    print(len(p1[i]))

                    logprob_chunks = model(cw_idxs, qw_idxs, candidates)

                    loss = F.nll_loss(logprob_chunks, chunk_y)
                    loss_val = loss.item()

                # Backward
                cand_loss.backward()
                nn.utils.clip_grad_norm_(cand_model.parameters(), args.max_grad_norm)
                cand_optimizer.step()
                cand_scheduler.step(step // batch_size)
                cand_ema(cand_model, step // batch_size)
                # might need a .detach here or something
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
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
                    cand_ema.assign(cand_model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                args.dev_eval_file,
                                                args.max_ans_len,
                                                args.use_squad_v2,
                                                cand_model)
                    saver.save(step, model, results[args.metric_name], device)
                    # also save candidate model?
                    cand_ema.resume(cand_model)
                    ema.resume(model)

                    # Log to console TODO
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard TODO
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
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)
            print(y1)

            # Forward
            if chunk:
                # TODO eval with actual candidate layer
                raise ValueError("No eval code yet")
                pass



                # TODO actual eval for scr here (eval candidate, scr seperately?)
                # log_p1, log_p2 = model(cw_idxs, qw_idxs)
                # y1, y2 = y1.to(device), y2.to(device)
                # loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                # nll_meter.update(loss.item(), batch_size)

                # # Get F1 and EM scores
                # p1, p2 = log_p1.exp(), log_p2.exp()
                # starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)
            else:
                log_p1, log_p2 = model(cw_idxs, qw_idxs)
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
