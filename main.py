"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import os
import pathlib

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, StochasticWeightAveraging, \
    BasePredictionWriter, LearningRateMonitor
from core.callbacks import MultiStageABAW5

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from core import config, ABAW5Model, ABAW5DataModule
from core.config import cfg
from core.io import pathmgr

from datetime import datetime
from tqdm import tqdm
import wandb
import numpy as np


class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: str = 'epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Make prediction folder
        predictions_postfix = 0
        while os.path.isdir(os.path.join(self.output_dir, "predictions_{}".format(predictions_postfix))):
            predictions_postfix += 1

        prediction_folder = os.path.join(self.output_dir, "predictions_{}".format(predictions_postfix))

        os.makedirs(prediction_folder, exist_ok=True)

        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        print('Saved file: ', os.path.join(self.output_dir, "predictions.pt"))
        print('Creating txt files...')
        preds = []
        # ytruths = []
        findexes = []
        video_ids = []
        for k in predictions[0]:
            preds.append(k[0])
            # ytruths.append(k[1])
            findexes.append(k[2])
            video_ids += k[3]

        if cfg.TASK == 'AU':
            preds = np.squeeze(1 * (torch.concat(preds).numpy() >= 0.5))
            header_name = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
        elif cfg.TASK == 'VA':
            preds = np.squeeze(torch.concat(preds).numpy().astype(float))
            header_name = ['valence', 'arousal']
        elif cfg.TASK == 'EXPR':
            preds = np.squeeze(torch.concat(preds).numpy())
            header_name = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        else:
            raise ValueError('Do not support write prediction for {} task'.format(cfg.TASK))

        # ytruths = np.squeeze(torch.concat(ytruths).numpy())
        findexes = torch.concat(findexes).numpy()
        video_ids = np.array(video_ids)

        video_id_uq = list(pd.unique(video_ids))
        num_classes = preds.shape[-1]
        print(prediction_folder)

        sample_prediction = pd.read_csv(f'/home/tu/ABAW3/testset/CVPR_5th_ABAW_{cfg.TASK}_test_set_example.txt')
        sample_prediction_indexes = np.array([x.split('/')[0] for x in sample_prediction['image_location'].values])

        all_prediction = []

        for vd in video_id_uq:
            num_sample_pred = np.sum(sample_prediction_indexes == vd)
            list_row = video_ids == vd
            list_preds = preds[list_row, :, :].reshape(-1, preds.shape[-1])
            list_indexes = findexes[list_row, :].reshape(-1)

            if np.sum(np.diff(list_indexes) < 0):
                print('Please check: {}. Indexes are not consistent'.format(vd))
            # Remove duplicate rows. Because we split sequentially => only padding at the end

            num_frames = num_sample_pred # len(np.unique(list_indexes))

            write_prediction = list_preds[:num_frames, :]
            write_prediction_index = np.array(['{}/{:05d}.jpg'.format(vd, idx+1) for idx in range(num_frames)]).reshape(-1, 1)

            cur_prediction = np.concatenate((write_prediction_index, write_prediction), axis=1)

            all_prediction.append(cur_prediction)

        all_prediction = np.concatenate(all_prediction)

        if cfg.TASK in ['AU', 'VA']:
            pd.DataFrame(data=all_prediction, columns=['image_location'] + header_name).to_csv(
                '{}/predictions.txt'.format(prediction_folder), index=False)
        elif cfg.TASK == 'EXPR':
            with open('{}/predictions.txt'.format(prediction_folder), 'w') as fd:
                fd.write(','.join(['image_location'] + header_name) + '\n')
                fd.write('\n'.join(all_prediction))
        else:
            raise ValueError('Do not support write prediction for {} task'.format(cfg.TASK))



if __name__ == '__main__':
    config.load_cfg_fom_args("abaw5 2023")
    config.assert_and_infer_cfg()
    cfg.freeze()

    pl.seed_everything(cfg.RNG_SEED)
    # Sets the internal precision of float32 matrix multiplications on NVIDIA 3090, disable if you want
    torch.set_float32_matmul_precision('high')

    pathmgr.mkdirs(cfg.OUT_DIR)
    run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.LOGGER == 'wandb' and cfg.TEST_ONLY == 'none':
        cfg_file_dir = pathlib.Path(cfg.OUT_DIR, '{}_{}'.format(cfg.TASK, run_version))
    else:
        cfg_file_dir = pathlib.Path(cfg.OUT_DIR, cfg.TASK, run_version)
    pathmgr.mkdirs(cfg_file_dir)
    cfg_file = config.dump_cfg(cfg_file_dir)

    if cfg.LOGGER == 'wandb' and not cfg.OPTIM.TUNE_LR and cfg.TEST_ONLY == 'none':
        logger = WandbLogger(project='Affwild2-abaw5', save_dir=cfg.OUT_DIR, name='{}_{}'.format(cfg.TASK, run_version),
                             offline=False)
        output_dir = cfg_file_dir

    else:
        # raise ValueError('Do not implement with {} logger yet.'.format(cfg.LOGGER))
        print('Use TensorBoard logger as default')
        logger = TensorBoardLogger(cfg.OUT_DIR, name=cfg.TASK, version=run_version)
        output_dir = logger.log_dir

    if cfg.TEST.WEIGHTS != '':
        result_dir = '/'.join(cfg.TEST.WEIGHTS.split('/')[:-1])
    else:
        result_dir = ''

    print('Working on Task: ', cfg.TASK)
    print(cfg.MODEL.BACKBONE, ' unfreeze: ', cfg.MODEL.BACKBONE_FREEZE)
    max_epochs = cfg.OPTIM.MAX_EPOCH if cfg.TEST.WEIGHTS == '' else 1

    abaw5_dataset = ABAW5DataModule()
    print('cfg.MODEL.USE_AUX',cfg.MODEL.USE_AUX)
    abaw5_model = ABAW5Model(do_mixup=False, use_aux=cfg.MODEL.USE_AUX)

    fast_dev_run = False
    richProgressBarTheme = RichProgressBarTheme(description="blue", progress_bar="green1",
                                                progress_bar_finished="green1")

    # backbone_finetunne = MultiStageabaw5(unfreeze_temporal_at_epoch=3, temporal_initial_ratio_lr=0.1,
    #                                         should_align=True, initial_denom_lr=10, train_bn=True)
    ckpt_cb = ModelCheckpoint(monitor='val_metric', mode="max", save_top_k=1, save_last=True)
    trainer_callbacks = [ckpt_cb,
                         PredictionWriter(output_dir=output_dir, write_interval='epoch'),
                         LearningRateMonitor(logging_interval=None)
                         ]
    if cfg.LOGGER in ['TensorBoard', 'none'] and not cfg.OPTIM.TUNE_LR:
        trainer_callbacks.append(RichProgressBar(refresh_rate=1, theme=richProgressBarTheme, leave=True))
        trainer_callbacks.append(RichModelSummary())

    if cfg.OPTIM.USE_SWA:
        swa_callbacks = StochasticWeightAveraging(swa_epoch_start=0.8,
                                                  swa_lrs=cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR,
                                                  annealing_epochs=1)
        trainer_callbacks.append(swa_callbacks)

    trainer = Trainer(accelerator='gpu', devices=1, accumulate_grad_batches=cfg.TRAIN.ACCUM_GRAD_BATCHES,
                      max_epochs=max_epochs, deterministic=True, callbacks=trainer_callbacks,
                      enable_model_summary=False,
                      num_sanity_val_steps=0, enable_progress_bar=True, logger=logger,
                      gradient_clip_val=None,
                      limit_train_batches=cfg.TRAIN.LIMIT_TRAIN_BATCHES, limit_val_batches=1.,
                      # limit_train_batches=0.05, limit_val_batches=0.05,
                      precision=32 // (cfg.TRAIN.MIXED_PRECISION + 1),
                      auto_lr_find=cfg.OPTIM.TUNE_LR,  # auto_scale_batch_size=None,
                      fast_dev_run=fast_dev_run,
                      )

    if cfg.TEST_ONLY != 'none':
        print('Testing only. Loading checkpoint: ', cfg.TEST_ONLY)
        print(cfg.TRANF.TARGET)
        if not os.path.isfile(cfg.TEST_ONLY):
            raise ValueError('Could not find {}'.format(cfg.TEST_ONLY))
        # Load pretrained weights
        pretrained_state_dict = torch.load(cfg.TEST_ONLY)['state_dict']
        abaw5_model.load_state_dict(pretrained_state_dict, strict=False)

        # Prepare test set
        abaw5_dataset.setup()
        # Re-evaluate validation set
        print('Re-evaluate validation set')
        trainer.test(dataloaders=abaw5_dataset.val_dataloader(), ckpt_path=None, model=abaw5_model)

        # Generate train prediction
        #print('Generate train prediction')
        #trainer.predict(dataloaders=abaw5_dataset.train_dataloader(shufflex=False), ckpt_path=None, model=abaw5_model)

        ## Generate val prediction
        #print('Generate val prediction')
        #trainer.predict(dataloaders=abaw5_dataset.val_dataloader(), ckpt_path=None, model=abaw5_model)

        # Generate test prediction
        print('Generate test prediction')
        trainer.predict(dataloaders=abaw5_dataset.test_dataloader(), ckpt_path=None, model=abaw5_model)
        print('Testing finished.')

    elif cfg.OPTIM.TUNE_LR:
        print('Auto LR Find')
        trainer.tune(abaw5_model, datamodule=abaw5_dataset, lr_find_kwargs={})
    else:
        #
        trainer.fit(abaw5_model, datamodule=abaw5_dataset)

        if cfg.LOGGER == 'wandb':
            wandb.run.log_code("./core/", include_fn=lambda path: path.endswith(".py") or path.endswith('.yaml'), )

        print('Pass with best val_metric: {}. Generating the prediction ...'.format(ckpt_cb.best_model_score))
        if cfg.OPTIM.USE_SWA:
            print('Evaluating with SWA')
            trainer.test(dataloaders=abaw5_dataset.val_dataloader(), ckpt_path=None, model=abaw5_model)
            trainer.save_checkpoint(ckpt_cb.last_model_path.replace('.ckpt', '_swa.ckpt'))

        trainer.test(dataloaders=abaw5_dataset.val_dataloader(), ckpt_path='best')
        trainer.predict(dataloaders=abaw5_dataset.test_dataloader(), ckpt_path='best')
