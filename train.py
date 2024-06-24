import math
import argparse
import pprint
from distutils.util import strtobool
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.strategies import DDPStrategy
# pytorch_lightning和pytorch对应版本问题 https://blog.csdn.net/sinat_41506268/article/details/132230225

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning_trainer.data import MultiSceneDataModule
from src.lightning_trainer.trainer import PL_Trainer

loguru_logger = get_rank_zero_only_logger(loguru_logger)

# n_gpus=4
# torch_num_workers=4
# batch_size=1
# pin_memory=true
def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, default="configs/megadepth_test_topicfmfast.py", help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default="megadepth_bs4-1-4")  # "outdoor-bs=$(($n_gpus * $batch_size))"
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default="pretrained/topicfm_fast.ckpt",
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=9000)
    parser.add_argument('--limit_val_batches', type=int, default=1)
    parser.add_argument('--num_sanity_val_steps', type=int, default=10)
    parser.add_argument('--benchmark', type=str, default=True)
    parser.add_argument('--max_epochs', type=int, default=30)

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    # config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation
    
    # scale lr and warmup-step automatically
    _n_gpus = setup_gpus(args.devices)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR  # * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_Trainer(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, epoch_start=args.epoch_start)
    loguru_logger.info(f"Model LightningModule initialized!")
    
    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"Model DataModule initialized!")
    
    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=10, mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_n_epochs=0,  # avoid repeated samples!
        # weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()