import os
from pathlib import Path
import warnings

import hydra
import torch
# import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
# from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from skimage.io import imsave

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    # from src.model.model_wrapper import ModelWrapper
    from src.model.model_wrapper_erp import ModelWrapperERP



def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)

    print(cyan(f"Saving outputs to {output_dir}."))
    # latest_run = output_dir.parents[1] / "latest-run"
    # os.system(f"rm {latest_run}")
    # os.system(f"ln -s {output_dir} {latest_run}")

    ###################logger#####################
    logger_cfg = {"save_dir": output_dir, "name": "tensorboard_logs"}
    logger = pl.loggers.TensorBoardLogger(**logger_cfg)

    ###################callbacks#####################
    callbacks = []
    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            # save_top_k=cfg.checkpointing.save_top_k,
            filename="{epoch:06}",
            verbose=True,
            save_last=True,
            # mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )

    # # Set up checkpointing.
    # callbacks.append(
    #     ModelCheckpoint(
    #         output_dir / "checkpoints",
    #         every_n_train_steps=cfg.checkpointing.every_n_train_steps,
    #         save_top_k=cfg.checkpointing.save_top_k,
    #         monitor="info/global_step",
    #         mode="max",  # save the lastest k ckpt, can do offline test later
    #     )
    # )

    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    class LitProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.set_description('running validation ...')
            return bar
        def init_train_tqdm(self):
            bar = super().init_train_tqdm()
            bar.set_description('running training ...')
            return bar
    bar = LitProgressBar()
    callbacks.append(bar)
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=True, #cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    
    # set log dir
    trainer.logdir = output_dir

    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    if cfg.mode == "train":
        # only load monodepth
        if cfg.model.encoder.pretrained_monodepth is not None:
            # strict_load = False
            # pretrained_model = torch.load(cfg.model.encoder.pretrained_monodepth, map_location='cpu')
            # if 'state_dict' in pretrained_model:
            #     pretrained_model = pretrained_model['state_dict']
            # import pdb;pdb.set_trace()
            # encoder.pretrained_monodepth_model.load_state_dict(pretrained_model, strict=strict_load)
            # print(
            #     cyan(
            #         f"Loaded pretrained monodepth: {cfg.model.encoder.pretrained_monodepth}"
            #     )
            # )
            strict_load = False
            pretrained_model = torch.load(cfg.model.encoder.pretrained_monodepth, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']
            msg = encoder.load_state_dict(pretrained_model, strict=strict_load)
            print(f"msg: {msg}")
            print(
                f"Loaded pretrained monodepth: {cfg.model.encoder.pretrained_monodepth}"
            )

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    if cfg.mode == "train":
        if (output_dir / "checkpoints" / "last.ckpt").exists():
            checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
            cfg.checkpointing.resume = True

    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights, without optimizer states
        # e.g., fine-tune from the released weights on other datasets
        model_wrapper = ModelWrapperERP.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=True)
        print(cyan(f"Loaded weigths from {checkpoint_path}."))

    else:
        model_wrapper = ModelWrapperERP(**model_kwargs)


    

    model_wrapper.image_dir = output_dir # used in output images during training
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=(
            checkpoint_path if cfg.checkpointing.resume else None))
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
