# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.model_utils import weights_init_kaiming
from fastreid.layers import *
import pdb
class BlackHead(nn.Module):
    def __init__(self, cfg, num_classes, in_feat, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self._num_classes = num_classes

        self.pool_layer = nn.Sequential(
            pool_layer,
            Flatten()
        )
        self.bnneck = NoBiasBatchNorm1d(in_feat)
        self.bnneck.apply(weights_init_kaiming)

        if cfg.MODEL.HEADS.CLS_LAYER == 'linear':
            self.classifier = nn.Linear(in_feat, self._num_classes, bias=False)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'arcface':
            self.classifier = Arcface(cfg, in_feat)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'circle':
            self.classifier = Circle(cfg, in_feat)
        else:
            self.classifier = nn.Linear(in_feat, self._num_classes, bias=False)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        # evaluation

        # training
        try:
            pred_class_logits = self.classifier(bn_feat)
        except TypeError:
            pred_class_logits = self.classifier(bn_feat, targets)
        return pred_class_logits, global_feat
