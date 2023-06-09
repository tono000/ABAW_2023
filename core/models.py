"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import sys

import torch.nn
from pytorch_lightning import LightningModule

from torchmetrics import F1Score, PearsonCorrCoef, MeanSquaredError
from torch.optim import lr_scheduler
from core.metrics import ConCorrCoef

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, resnet18, resnet50, \
    resnet34,efficientnet_v2_s
from torch import nn
from torch.nn import functional as F

from core.config import cfg
from core.loss import CCCLoss, CELogitLoss, BCEwithLogitsLoss, MSELoss, SigmoidFocalLoss, CEFocalLoss
from core.mixup import MixupTransform

from core.vivit_module import PreNorm, Attention, FeedForward

from pretrained import vggface2
from pretrained.facex_zoo import get_facex_zoo
from functools import partial
import math

from einops import rearrange, repeat


# Facenet https://github.com/timesler/facenet-pytorch

def get_vggface2(model_name):
    if 'senet50' in model_name:
        vgg2_model = vggface2.resnet50(include_top=False, se=True, num_classes=8631)
        vgg2_ckpt = torch.load('pretrained/vggface2_weights/senet50_ft_weight.pth')
    elif 'resnet50' in model_name:
        vgg2_model = vggface2.resnet50(include_top=False, se=False, num_classes=8631)
        vgg2_ckpt = torch.load('pretrained/vggface2_weights/resnet50_ft_weight.pth')
    else:
        raise ValueError('Unkown model name {} for VGGFACE2'.format(model_name))

    vgg2_model.load_state_dict(vgg2_ckpt['model_state_dict'])
    return vgg2_model


class Transformer(nn.Module):
    # Based on https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, use_torch_attn=True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SpatialProj(nn.Module):
    def __init__(self, num_outputs, num_feat, dropout=0., expand=1.5):
        super(SpatialProj, self).__init__()
        if expand <= 1.:
            self.proj = nn.Sequential(nn.Linear(num_feat, num_feat), nn.ReLU())
        else:
            self.proj = nn.Sequential(nn.Linear(num_feat, int(num_feat * expand)), nn.ReLU(),
                                      nn.Linear(int(num_feat * expand), num_feat), nn.ReLU())
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc_pred = nn.Linear(num_feat, num_outputs)

    def forward(self, x, return_proj=True):
        # Input x: [b t n (h w)]
        x = rearrange(x, 'b t n m -> b t m n')
        proj = self.proj(x)
        if self.dropout is not None:
            x = self.dropout(proj)
        else:
            x = proj

        x = torch.mean(x, dim=2)
        if return_proj:
            return self.fc_pred(x), rearrange(proj, 'b t m n -> b t n m')
        else:
            return x


class TransformerProj(nn.Module):
    # Based on https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py#L28
    def __init__(self, num_outputs, seq_len, num_patches=16, num_feat=512, nhead=8, depth=8, dim_feed=256, dropout=0.1,
                 pool=None):
        super(TransformerProj, self).__init__()
        # Input should be [b s n (h w)]
        # Transform [b s n (h w)] to [b s (h w)  n]

        assert pool in {'cls', 'mean', None}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_frames = seq_len
        num_patches = num_patches
        dim = num_feat
        scale_dim = 4

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches+1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, nhead, dim_feed, dim_feed, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, nhead, dim_feed, dim_feed, dropout)

        self.dropout = nn.Dropout(dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_outputs)
        )

    def forward(self, x, return_middle=False):
        # src: b t n (h w)
        x = rearrange(x, 'b t n p -> b t p n')
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        if self.pool is not None:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # else:
        #     x = x[:, 1:]
        if return_middle:
            return x, self.mlp_head(x)
        else:
            return self.mlp_head(x)


def get_backbone(backbone_name, backbone_freeze=None, backbone_freeze_bn=True, aux_feat=None):
    backbone_dict = {'regnet-400mf': (regnet_y_400mf, 48, 104, 208, 440),
                     'regnet-800mf': (regnet_y_800mf, 64, 144, 320, 784),
                     'regnet-1.6gf': (regnet_y_1_6gf, 48, 120, 336, 888),
                     'regnet-3.2gf': (regnet_y_3_2gf, 96, 192, 432, 1512),
                     'resnet18': (resnet18, 64, 128, 256, 512), 'resnet34': (resnet34, 64, 128, 256, 512),
                     'resnet50': (resnet50, 256, 512, 1024, 2048),
                     'efficientnet_v2_s': (efficientnet_v2_s, 1, 1, 1, 1280)}
    if backbone_name not in backbone_dict:
        raise ValueError(
            'Do not support {} backbone. Only support backbone in {} with ImageNet pretrained'.format(backbone_name,
                                                                                                      backbone_dict.keys()))

    bb_model = backbone_dict[backbone_name][0](weights='DEFAULT')

    if aux_feat is not None and isinstance(aux_feat, (list, tuple)):
        train_return_nodes = {}
        for aux_name in aux_feat:
            train_return_nodes.update({aux_name: '{}_aux'.format(aux_name)})
        train_return_nodes.update({'trunk_output.block4.block4-5.activation': 'feat'})
        val_return_nodes = {'trunk_output.block4.block4-5.activation': 'feat'}
        return_nodes = None
    else:
        return_nodes = {'trunk_output.block4.block4-5.activation': 'feat'}
        train_return_nodes = None
        val_return_nodes = None

    backbone = create_feature_extractor(bb_model, return_nodes=return_nodes, train_return_nodes=train_return_nodes,
                                        eval_return_nodes=val_return_nodes)
    num_feats = backbone_dict[backbone_name][1:]

    # Freeze backbone model
    if len(backbone_freeze) > 0:
        for named, param in backbone.named_parameters():
            do_freeze = True
            if 'all' not in backbone_freeze or not (
                    isinstance(param, nn.BatchNorm2d) and backbone_freeze_bn):
                for layer_name in backbone_freeze:
                    if layer_name in named:
                        do_freeze = False
                        break
            if do_freeze:
                param.requires_grad = False

    return backbone, num_feats


class ABAW5BaseModel(nn.Module):
    def __init__(self, bacbone, backbone_freeze=None, backbone_freeze_bn=True, aux_feat=None, transf_nhead=8,
                 transf_num_enc=2, seq_len=1,
                 transf_dimfc=256, transf_dropout=0.1, transf_norm_first=True,
                 aux_coeff=0.2, num_outputs=2):
        super(ABAW5BaseModel, self).__init__()
        self.num_outputs = num_outputs
        self.seq_len = seq_len

        self.backbone, self.num_feats = get_backbone(bacbone, backbone_freeze, backbone_freeze_bn, aux_feat)

        num_feat_spatial = self.num_feats[-1]

        self.temporal_head = TransformerProj(num_outputs=self.num_outputs, seq_len=seq_len, num_feat=num_feat_spatial,
                                             nhead=transf_nhead, depth=transf_num_enc,
                                             dim_feed=transf_dimfc, dropout=transf_dropout,
                                             pool=None)
        self.aux_coeff = aux_coeff
        if self.aux_coeff > 0.:
            # Default expand is 1
            self.spatial_aux = SpatialProj(num_outputs=self.num_outputs, num_feat=num_feat_spatial, dropout=0.,
                                           expand=1.5)

            self.temporal_head_mix = TransformerProj(num_outputs=self.num_outputs, seq_len=seq_len,
                                                     num_feat=num_feat_spatial,
                                                     nhead=transf_nhead, depth=transf_num_enc,
                                                     dim_feed=transf_dimfc, dropout=transf_dropout,
                                                     pool=None)

    def forward(self, x, is_train, indexes=None):
        """

        :param x: 5-D vector, batch_size x seq_len x n_channels x H x W
        :return:
        """
        num_seq = x.shape[0]

        feat = rearrange(x, 'b s n h w -> (b s) n h w ')
        backbone_feats = self.backbone(feat)  # [batch size * seq] x num_feat x 4 x 4

        feat = backbone_feats['feat']

        # Convert to batch size x seq x num_feat
        feat = rearrange(feat, '(b s) n h w -> b s n (h w)', b=num_seq, s=self.seq_len)

        out_auxes = []
        if self.aux_coeff > 0.:  # and is_train:
            # spatial_aux is transformer, add 1 dimension as length => batch size * seq x 1 x num_feat
            out_aux, spatial_proj = self.spatial_aux(feat, return_proj=True)
            out_auxes.append(out_aux)

        else:
            spatial_proj = None

        transf_proj, out = self.temporal_head(feat, return_middle=True)

        if self.aux_coeff > 0.:
            out_mix = self.temporal_head_mix(spatial_proj)
            out_auxes.append(out_mix)

        # print('out shape: ', out.shape)
        # sys.exit(0)
        return out, out_auxes


class ABAW5Model(LightningModule):

    def __init__(self, do_mixup=False, use_aux=0.2):
        # TODO: Load backbone pretrained on static frames
        super(ABAW5Model, self).__init__()
        self.seq_len = cfg.DATA_LOADER.SEQ_LEN
        self.task = cfg.TASK
        self.scale_factor = 1.
        self.threshold = 0.5
        self.learning_rate = cfg.OPTIM.BASE_LR
        self.backbone_name = cfg.MODEL.BACKBONE
        self.backbone_freeze = cfg.MODEL.BACKBONE_FREEZE
        self.aux_coeff = use_aux

        # Set metrics and criterion
        self.get_metrics_criterion()

        # Create model
        num_enc_dec = cfg.TRANF.NUM_ENC_DEC
        self.model = ABAW5BaseModel(bacbone=cfg.MODEL.BACKBONE, backbone_freeze=cfg.MODEL.BACKBONE_FREEZE,
                                    backbone_freeze_bn=cfg.MODEL.FREEZE_BATCHNORM,
                                    aux_feat=None, aux_coeff=self.aux_coeff, seq_len=self.seq_len,
                                    transf_nhead=cfg.TRANF.NHEAD, transf_num_enc=num_enc_dec,
                                    transf_dimfc=cfg.TRANF.DIM_FC, transf_dropout=cfg.TRANF.DROPOUT,
                                    transf_norm_first=True,
                                    num_outputs=self.num_outputs)
        self.fusion_strategy = 0
        if self.aux_coeff > 0.:
            self.fusion_strategy = cfg.MODEL.FUSION_STRATEGY
            num_fusion = 3

            if self.fusion_strategy == 1:
                num_dim = num_fusion * self.num_outputs
                self.fc_fusion = nn.Sequential(nn.Linear(num_dim, num_dim), nn.ReLU(),
                                               nn.Linear(num_dim, self.num_outputs))
            elif self.fusion_strategy == 2:
                self.fc_fusion = nn.Sequential(nn.Linear(num_fusion, num_fusion), nn.Tanh(),
                                               nn.Linear(num_fusion, num_fusion), nn.Sigmoid())
            else:
                raise ValueError('Do not support fusion_strategy {} with aux_coeff.'.format(self.fusion_strategy))

        else:
            self.fusion_strategy = -1

        self._reset_parameters()
        if cfg.MODEL.BACKBONE_PRETRAINED != 'none':
            pretrained_state_dict = torch.load(cfg.MODEL.BACKBONE_PRETRAINED)['state_dict']
            self.load_state_dict(pretrained_state_dict, strict=False)
            # Freeze backbone
            self.model.backbone.requires_grad_(False)
            # Freeze temporal head transf
            self.model.temporal_head.tranf.requires_grad_(False)
            # Freeze temporal head fc
            self.model.temporal_head.fc_head.requires_grad_(False)

    def forward(self, batch):

        # batch['image'] batch size x seq x 3 x h x w
        use_indexes = torch.unsqueeze(batch['index'], dim=-1)

        out, out_aux = self.model(batch['image'], is_train=self.training, indexes=use_indexes)

        return out, out_aux

    def _shared_eval(self, batch, batch_idx, cal_loss=False):
        out, out_aux = self(batch)

        if len(out_aux) > 0:
            if self.fusion_strategy == 1:
                out_fusion = torch.concat([out, ] + out_aux, dim=-1)
                out_fusion = self.fc_fusion(out_fusion)

            else:
                out_fusion = torch.stack([out, ] + out_aux, dim=-1)
                out_fusion_weight = self.fc_fusion(out_fusion)
                out_fusion = torch.sum(out_fusion * out_fusion_weight, dim=-1)

        else:
            out_fusion = None
        loss = None
        if cal_loss:
            if self.task != 'MTL':
                if loss is None:
                    loss = self.loss_func(out, batch[self.task])
                    if out_fusion is not None:
                        loss = loss * self.aux_coeff + self.loss_func(out_fusion, batch[self.task])
                        out = out_fusion

                if self.aux_coeff:
                    for aux in out_aux:
                        loss += self.aux_coeff * self.loss_func(aux, batch[self.task])
                    loss /= (1 + (1 + len(out_aux)) * self.aux_coeff)

        return out, loss

    def update_metric(self, out, y, is_train=True):
        if self.task == 'EXPR':
            y = torch.reshape(y, (-1,))
            # out = F.softmax(out, dim=1)
        elif self.task == 'AU':
            out = torch.sigmoid(out)
            y = torch.reshape(y, (-1, self.num_outputs))

        elif self.task == 'VA':
            y = torch.reshape(y, (-1, self.num_outputs))

        out = torch.reshape(out, (-1, self.num_outputs))

        if is_train:
            self.train_metric(out, y)
        else:
            self.val_metric(out, y)

    def training_step(self, batch, batch_idx):

        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=True)

        self.log('train_metric', self.train_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=cfg.TRAIN.BATCH_SIZE)

        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=False)

        self.log_dict({'val_metric': self.val_metric, 'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
                      batch_size=cfg.TEST.BATCH_SIZE)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=False)

        if self.task != 'MTL':
            if self.task == 'EXPR':
                out = torch.argmax(F.softmax(out, dim=-1), dim=-1, keepdim=True)
            elif self.task == 'AU':
                out = torch.sigmoid(out)

            return out, batch[self.task], batch['index'], batch['video_id']
        else:
            raise ValueError('Do not implement MTL task.')

    def test_step(self, batch, batch_idx):
        # Copy from validation step
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=False)

        self.log_dict({'test_metric': self.val_metric, 'test_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
                      batch_size=cfg.TEST.BATCH_SIZE)

    def configure_optimizers(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_training_steps = self.num_steps_per_epoch()
        print('Number of training steps: ', self.num_training_steps)

        if cfg.OPTIM.NAME == 'adam':
            print('Adam optimization ', self.learning_rate)
            opt = torch.optim.Adam(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        elif cfg.OPTIM.NAME == 'adamw':
            print('AdamW optimization ', self.learning_rate)
            opt = torch.optim.AdamW(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        else:
            print('SGD optimization ', self.learning_rate)
            opt = torch.optim.SGD(model_parameters, lr=self.learning_rate, momentum=cfg.OPTIM.MOMENTUM,
                                  dampening=cfg.OPTIM.DAMPENING, weight_decay=cfg.OPTIM.WEIGHT_DECAY)

        opt_lr_dict = {'optimizer': opt}
        lr_policy = cfg.OPTIM.LR_POLICY
        if lr_policy == 'cos':
            # warmup_start_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            # scheduler = pl_lr_scheduler.LinearWarmupCosineAnnealingLR(opt, warmup_epochs=cfg.OPTIM.WARMUP_EPOCHS,
            #                                                           max_epochs=cfg.OPTIM.MAX_EPOCH,
            #                                                           warmup_start_lr=warmup_start_lr,
            #                                                           eta_min=cfg.OPTIM.MIN_LR)
            # opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched'}})
            pass

        elif lr_policy == 'cos-restart':
            min_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            t_0 = cfg.OPTIM.WARMUP_EPOCHS * self.num_training_steps
            print('Number of training steps: ', t_0 // cfg.OPTIM.WARMUP_EPOCHS)

            scheduler = DecayCosineANWRLR(opt, T_0=t_0, T_mult=1, eta_min=min_lr,
                                          T_trigger=16 * self.num_training_steps,
                                          decay_factor=0.1)

            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'cyclic':
            base_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            step_size_up = self.num_training_steps * cfg.OPTIM.WARMUP_EPOCHS // 2
            mode = 'triangular2'  # triangular, triangular2, exp_range
            scheduler = lr_scheduler.CyclicLR(opt, base_lr=base_lr, max_lr=self.learning_rate,
                                              step_size_up=step_size_up, mode=mode, gamma=1.,
                                              cycle_momentum=(cfg.OPTIM.NAME == 'sgd'))
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'cyclic-cos':
            base_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            tmax = self.num_training_steps * cfg.OPTIM.WARMUP_EPOCHS // 2
            scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=base_lr)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'reducelrMetric':
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10, min_lr=1e-7, mode='max')
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched',
                                                 "monitor": "val_metric"}})
        else:
            # TODO: add 'exp', 'lin', 'steps' lr scheduler
            pass
        return opt_lr_dict

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def num_steps_per_epoch(self) -> int:
        """Total training steps inferred from dataloaders and distributed setup."""
        # infinite training
        if self.trainer.max_epochs == -1 and self.trainer.max_steps == -1:
            return float("inf")

        if self.trainer.train_dataloader is None:
            print("Loading `train_dataloader` to estimate number of training steps.")
            self.trainer.reset_train_dataloader()

        total_batches = self.trainer.num_training_batches

        # iterable dataset
        if total_batches == float("inf"):
            return self.trainer.max_steps

        self.trainer.accumulate_grad_batches = self.trainer.accumulation_scheduler.get_accumulate_grad_batches(
            self.trainer.current_epoch)
        effective_batch_size = self.trainer.accumulate_grad_batches
        max_estimated_steps_per_epoch = math.ceil(total_batches / effective_batch_size)
        return max_estimated_steps_per_epoch

    def _reset_parameters(self) -> None:
        # Performs ResNet-style weight initialization
        for m_name, m in self.named_modules():
            if 'backbone' in m_name:
                continue
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_metrics_criterion(self):
        if self.task == 'VA':
            # Regression
            self.num_outputs = 2
            self.loss_func = partial(CCCLoss, scale_factor=self.scale_factor)

            self.train_metric = ConCorrCoef(num_classes=self.num_outputs)
            self.val_metric = ConCorrCoef(num_classes=self.num_outputs)

            # 2 x NUM_BINS (e.g., NUM_BINS=10, np.histogram(a, bins=10, range=(-1., 1.))
            # First row: valence, second row: arousal
            self.cls_weights = nn.Parameter(
                torch.tensor([[3.75464546, 1.89663824, 2.57556784, 2.98841786, 3.19120533, 0.27236339,
                               0.43483318, 0.87417645, 1.58016961, 2.36916838],
                              [5.94334483e+02, 4.30892500e+02, 9.76526912e+01, 7.24189076e+01,
                               1.30920623e+01, 2.63744453e-01, 3.75882148e-01, 6.66913017e-01,
                               9.21966354e-01, 1.16343447e+00]], requires_grad=False),
                requires_grad=False) if cfg.TRAIN.LOSS_WEIGHTS else None
        elif self.task == 'EXPR':
            # Classification
            self.num_outputs = 8
            self.label_smoothing = cfg.TRAIN.LABEL_SMOOTHING
            # Class weights
            self.cls_weights = nn.Parameter(torch.tensor(
                [0.42715146, 5.79871879, 6.67582676, 4.19317243, 1.01682121, 1.38816715, 2.87961987, 0.32818288],
                requires_grad=False), requires_grad=False) if cfg.TRAIN.LOSS_WEIGHTS else None

            # self.loss_func = partial(CELogitLoss, scale_factor=self.scale_factor, num_classes=self.num_outputs,
            #                          label_smoothing=self.label_smoothing, class_weights=self.cls_weights)
            self.loss_func = partial(CEFocalLoss, scale_factor=self.scale_factor, num_classes=self.num_outputs,
                                     label_smoothing=self.label_smoothing,
                                     alpha=cfg.OPTIM.FOCAL_ALPHA, gamma=cfg.OPTIM.FOCAL_GAMMA)

            self.train_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs, average='macro',
                                        task='multiclass')
            self.val_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs, average='macro',
                                      task='multiclass')

        elif self.task == 'AU':
            # Multi-label classification
            self.num_outputs = 12
            # Class weight for positive samples
            self.cls_weights = nn.Parameter(torch.tensor(
                [0.5945361, 0.90956106, 0.54041032, 0.33225202, 0.20905073, 0.23766536, 0.32939766, 2.78890182,
                 3.3465817, 2.84659439, 0.12843037, 0.7860732], requires_grad=False),
                requires_grad=False) if cfg.TRAIN.LOSS_WEIGHTS else None
            # self.loss_func = partial(BCEwithLogitsLoss, scale_factor=self.scale_factor, num_classes=self.num_outputs,
            #                         pos_weights=self.cls_weights)
            self.loss_func = partial(SigmoidFocalLoss, scale_factor=self.scale_factor, num_classes=self.num_outputs,
                                     alpha=cfg.OPTIM.FOCAL_ALPHA, gamma=cfg.OPTIM.FOCAL_GAMMA)

            self.train_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs,
                                        num_labels=self.num_outputs, average='macro', task='multilabel')
            self.val_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs,
                                      num_labels=self.num_outputs, average='macro', task='multilabel')

        elif self.task == 'MTL':
            raise ValueError('Do not support MTL at this time.')
        else:
            raise ValueError('Do not know {}'.format(self.task))


class DecayCosineANWRLR(lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, T_trigger=-1,
                 decay_factor=0.1, decay_etamin=0.1):
        super(DecayCosineANWRLR, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
        self.T_trigger = T_trigger
        self.decay_factor = decay_factor
        self.decay_etamin = decay_etamin
        print(T_trigger, decay_factor)

    def get_lr(self):
        if 0 < self.T_trigger <= self.last_epoch + 1:
            # print('Trigger decay factor')
            decay_factor = self.decay_factor
            decay_etamin = self.decay_etamin
        else:
            decay_factor = 1.
            decay_etamin = 1.

        return [self.eta_min * decay_etamin + (base_lr * decay_factor - self.eta_min * decay_etamin) * (
                1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
