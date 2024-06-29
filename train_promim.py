import os
import shutil
import argparse
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from src.utils.data import PPIRefPaddingCollate
from src.utils.train import *
from src.datasets.ppiref50k import get_PPIRef50K_dataset
from src.models.promim import ProMIM

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
log_rank = 0

import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_promim')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world_size', default=4, type=int, help='world size for distributed training')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_entity', type=str, default='', help='your wandb username')
    parser.add_argument('--wandb_dir', type=str, default='./logs_promim')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # Load configs
    config, config_name = load_config(args.config) 
    seed_all(config.train.seed)

    if dist.get_rank() == log_rank:
        # wandb setting
        if args.wandb:
            os.makedirs(args.wandb_dir, exist_ok=True)
            run = wandb.init(
                entity=args.wandb_entity,
                project="train_promim", 
                name=args.tag,
                config=dict(config),
                dir=args.wandb_dir
            )

        # Logging
        if args.debug:
            logger = get_logger('train', None)
            writer = BlackHole()
        else:
            if args.resume:
                log_dir = get_new_log_dir(args.logdir, prefix=config_name+'-resume', tag=args.tag)
            else:
                log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
            ckpt_dir = os.path.join(log_dir, 'checkpoints')
            if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
            logger = get_logger('train', log_dir)
            writer = torch.utils.tensorboard.SummaryWriter(log_dir)
            tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
            if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
                shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        logger.info(args)
        logger.info(config)

    # Data
    if dist.get_rank() == log_rank: logger.info('Loading datasets...')
    # train
    train_dataset = get_PPIRef50K_dataset(config.data.train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PPIRefPaddingCollate(max_length=config.train.padding_size), sampler=train_sampler, drop_last=True)
    train_iterator = inf_iterator(train_loader)
    # val
    val_dataset = get_PPIRef50K_dataset(config.data.val)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PPIRefPaddingCollate(max_length=config.train.padding_size), sampler=val_sampler, drop_last=True)
    if dist.get_rank() == log_rank: logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))                                                          

    # Model
    if dist.get_rank() == log_rank: logger.info('Building model...')
    model = ProMIM(config.model)

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        if dist.get_rank() == log_rank: logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=f'cuda:{args.local_rank}')
        it_first = ckpt['iteration'] + 1
        lsd_result = model.load_state_dict(ckpt['model'], strict=False)

        if dist.get_rank() == log_rank: logger.info('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
        if dist.get_rank() == log_rank: logger.info('Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True) 
    if dist.get_rank() == log_rank: logger.info('Number of parameters: %d' % count_parameters(model))

    total_bsz = args.world_size * config.train.batch_size
    train_num = len(train_dataset)
    num_step_each_epoch = train_num // total_bsz

    def train(it):
        time_start = current_milli_time()
        model.train()

        # Shuffle
        if it > 1 and it % num_step_each_epoch == 1:
            epoch = it // num_step_each_epoch
            train_sampler.set_epoch(epoch)

        # Prepare data
        batch = next(train_iterator)

        # Forward pass
        loss_dict, metric_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # all_reduce loss & metric 
        def reduce_tensor(tensor):
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= args.world_size
            return rt
        torch.distributed.barrier()
        # all_reduce_loss
        reduced_loss_dict = {}
        for key, module_loss in loss_dict.items():
            reduced_module_loss = reduce_tensor(module_loss).clone()
            reduced_loss_dict[key] = reduced_module_loss
        reduced_loss = reduce_tensor(loss)
        # all_reduce_metric
        reduced_metric_dict = {}
        for key, metric in metric_dict.items():
            reduced_metric = reduce_tensor(metric).clone()
            reduced_metric_dict[key] = reduced_metric

        # Logging
        if dist.get_rank() == log_rank:
            scalar_dict = {}
            scalar_dict.update({
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })
            for key, reduced_metric in reduced_metric_dict.items():
                scalar_dict[key] = reduced_metric.item()
            log_losses(reduced_loss, reduced_loss_dict, scalar_dict, it=it, tag='train', logger=logger, writer=writer)

            if args.wandb:
                wandb_scalar_dict = {}
                wandb_loss_dict = {}
                for key, value in scalar_dict.items():
                    wandb_scalar_dict[f'train/{key}'] = value
                for key, value in reduced_loss_dict.items():
                    wandb_loss_dict[f'train/{key}'] = value
                wandb.log(wandb_scalar_dict, step=it)
                wandb.log(wandb_loss_dict, step=it)

    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc=f'Validate{args.local_rank}', dynamic_ncols=True)):

                # Forward pass
                loss_dict, metric_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')
                for key, metric in metric_dict.items():
                    scalar_accum.add(name=key, value=metric, batchsize=None, mode=None)
                for k, l in loss_dict.items():
                    scalar_accum.add(name=k, value=l.reshape(1), batchsize=None, mode=None)


        def reduce_tensor(tensor):
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= args.world_size
            return rt
        avg_loss = scalar_accum.get_average('loss')
        avg_metric_dict = {}
        for key, _ in metric_dict.items():
            avg_metric_dict[f'avg_{key}'] = scalar_accum.get_average(key)
        avg_loss_dict = {}
        for key, _ in loss_dict.items():
            avg_loss_dict[f'avg_{key}'] = scalar_accum.get_average(key)


        torch.distributed.barrier()
        reduced_avg_loss = reduce_tensor(torch.tensor(avg_loss).to(args.local_rank))

        reduced_avg_metric_dict = {}
        for key, _ in metric_dict.items():
            reduced_avg_metric_dict[f'reduced_avg_{key}'] = reduce_tensor(torch.tensor(avg_metric_dict[f'avg_{key}']).to(args.local_rank))
        
        reduced_avg_loss_dict = {}
        for key, _ in loss_dict.items():
            reduced_avg_loss_dict[f'reduced_avg_{key}'] = reduce_tensor(torch.tensor(avg_loss_dict[f'avg_{key}']).to(args.local_rank))

        if dist.get_rank() == log_rank:
            writer.add_scalar('val/loss', reduced_avg_loss.item(), it)
            for key, _ in metric_dict.items():
                writer.add_scalar(f'val/{key}', reduced_avg_metric_dict[f'reduced_avg_{key}'].item(), it)
            for key, _ in loss_dict.items():
                writer.add_scalar(f'val/{key}', reduced_avg_loss_dict[f'reduced_avg_{key}'].item(), it)                
            logstr = '[%s] Iter %05d' % ('val', it)
            logger.info(logstr)

            if args.wandb:
                wandb_metric_dict = {}
                wandb_loss_dict = {}
                for key, _ in metric_dict.items():
                    wandb_metric_dict[f'val/{key}'] = reduced_avg_metric_dict[f'reduced_avg_{key}'].item()
                for key, _ in loss_dict.items():
                    wandb_loss_dict[f'val/{key}'] = reduced_avg_loss_dict[f'reduced_avg_{key}'].item()
                wandb.log(wandb_metric_dict, step=it)
                wandb.log(wandb_loss_dict, step=it)

        # Trigger scheduler
        if it != it_first:
            if config.train.scheduler.type == 'plateau':
                scheduler.step(reduced_avg_loss.item())
            else:
                scheduler.step()
        return reduced_avg_loss.item()

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if (not args.debug) and (dist.get_rank() == log_rank):
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
