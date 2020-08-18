# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy

import torch
from torch import nn
import pdb
from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..backbones.resnet import Bottleneck
from ..heads import build_reid_heads
from ..model_utils import weights_init_kaiming
from ...layers import GeneralizedMeanPoolingP, Flatten
from .STN import *
from projects.Black_reid.build_losses import reid_losses, iou_losses
from projects.Black_reid.blackhead import BlackHead

@META_ARCH_REGISTRY.register()
class HAA_BASELINE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # backbone
        resnet = build_backbone(cfg)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())
        
        self.bone = nn.Sequential(copy.deepcopy(self.backbone), copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))

        if cfg.MODEL.HEADS.POOL_LAYER == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        else:
            pool_layer = nn.Identity()
        self.bone_pool = self._build_pool_reduce(copy.deepcopy(pool_layer),2048,1536)
        self.heads = build_reid_heads(cfg, 1536,  nn.Identity())      

        #Adaptive Attention Branch
        self.classifier_black = BlackHead(cfg, 2, 1536, nn.Identity()) 
        self.prob = BlackHead(cfg, 2, 2, nn.Identity())
        self.final_head = build_reid_heads(cfg, 3072, nn.Identity())
        
        
        #HSA_branch
        self.stn = SpatialTransformBlock(1,24,512)
        model_dict =self.stn.state_dict() 
        pretrained_dict = torch.load(self._cfg.DATASETS.STN_ROOT)
        pretrained_dict =  {k[4:]: v for k, v in pretrained_dict.items() if k.startswith('stn')} 
        self.stn.load_state_dict(pretrained_dict)  
        for param in self.stn.parameters():
            param.requires_grad = False
        self.hsa = nn.Sequential(copy.deepcopy(self.backbone), copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.hsa_1_pool = self._build_pool_reduce(copy.deepcopy(pool_layer),2048,512)
        self.hsa_1_head = build_reid_heads(cfg, 512, nn.Identity())
        self.hsa_2_pool = self._build_pool_reduce(copy.deepcopy(pool_layer),2048,512)
        self.hsa_2_head = build_reid_heads(cfg, 512, nn.Identity())
        self.hsa_3_pool = self._build_pool_reduce(copy.deepcopy(pool_layer),2048,512)
        self.hsa_3_head = build_reid_heads(cfg, 512, nn.Identity())
        self.hsa_head = build_reid_heads(cfg, 1536, nn.Identity())
        
        #HAN
        self.han_pool_1 = copy.deepcopy(pool_layer)
        self.han_c_att_1 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(), nn.Linear(1024, 2048, bias=False))
        self.han_c_att_1[0].apply(weights_init_kaiming)
        self.han_c_att_1[1].apply(weights_init_kaiming)
        self.han_pool_2 = copy.deepcopy(pool_layer)
        self.han_c_att_2 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(), nn.Linear(1024, 2048, bias=False))
        self.han_c_att_2[0].apply(weights_init_kaiming)
        self.han_c_att_2[1].apply(weights_init_kaiming)
        self.han_pool_3 = copy.deepcopy(pool_layer)
        self.han_c_att_3 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(), nn.Linear(1024, 2048, bias=False))
        self.han_c_att_3[0].apply(weights_init_kaiming)
        self.han_c_att_3[1].apply(weights_init_kaiming)    

    def _build_pool_reduce(self, pool_layer, input_dim=2048, reduce_dim=256):
        pool_reduce = nn.Sequential(
            pool_layer,
            nn.Conv2d(input_dim, reduce_dim, 1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(True),
            Flatten()
        )
        pool_reduce.apply(weights_init_kaiming)
        return pool_reduce

    def forward(self, inputs):
        images = inputs["images"]
        targets = inputs["targets"]
        if self.training:
            blackid = inputs["black_id"]

        if not self.training:
            pred_feat = self.inference(images)
            return pred_feat, targets, inputs["camid"]

        features = self.bone(images)  # (bs, 2048, 16, 8)
        global_feat = self.bone_pool(features)
        logits, global_feat = self.heads(global_feat, targets)       

        #HSA branch
        stn = self.stn(images)
        head = stn[0][0]
        grid_list = stn[1][0]
        head_feat = self.hsa(head)
        hf_1 = self.HAN_1(head_feat[:,:,0:8,:])   
        hf1_pool_feat = self.hsa_1_pool(hf_1)
        hf1_logits, hf1_pool_feat = self.hsa_1_head(hf1_pool_feat, targets)   
        hf_2 = self.HAN_2(head_feat[:,:,8:16,:])
        hf2_pool_feat = self.hsa_2_pool(hf_2)
        hf2_logits, hf2_pool_feat = self.hsa_2_head(hf2_pool_feat, targets) 
        hf_3 = self.HAN_3(head_feat[:,:,16:24,:])
        hf3_pool_feat = self.hsa_3_pool(hf_3)
        hf3_logits, hf3_pool_feat = self.hsa_3_head(hf3_pool_feat, targets) 
        hf = torch.cat((hf1_pool_feat,hf2_pool_feat,hf3_pool_feat),dim=1)
        hf_logits, hf = self.hsa_head(hf, targets)
        
        #Adaptive Attention Branch
        gf = global_feat      
        black_logits, b1_pool_feat = self.classifier_black(gf)
        w, black_logits = self.prob(black_logits)
        w1 = torch.exp(w[:,0]) /(torch.exp(w[:,0])+torch.exp(w[:,1]))
        w2 = torch.exp(w[:,1]) /(torch.exp(w[:,0])+torch.exp(w[:,1]))
        w1 = w1.view(-1,1)
        w2 = w2.view(-1,1)        
        pred_feat = torch.cat((w1*gf,w2*hf),dim=1)
        pred_logits, pred_feat = self.final_head(pred_feat, targets) 


        return (logits, hf1_logits, hf2_logits, hf3_logits, hf_logits, pred_logits, black_logits), \
               (global_feat, hf1_pool_feat, hf2_pool_feat, hf3_pool_feat, hf, pred_feat), \
               targets, \
               grid_list, \
               blackid



    def losses(self, outputs, iters=0):
        loss_dict = {} 
        if iters<=int(int(self._cfg.SOLVER.MAX_ITER)*2/3):
            loss_dict.update(reid_losses(self._cfg, outputs[0][0], outputs[1][0], outputs[2],0.2, 0.2,'gf_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][1], outputs[1][1], outputs[2],0.2, 0.2,'hf1_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][2], outputs[1][2], outputs[2],0.2, 0.2,'hf2_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][3], outputs[1][3], outputs[2],0.2, 0.2,'hf3_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][4], outputs[1][4], outputs[2],0.2, 0.2,'hf_'))                
        else: 
            loss_dict.update(reid_losses(self._cfg, outputs[0][0], outputs[1][0], outputs[2], 0.143, 0.167, 'gf_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][1], outputs[1][1], outputs[2], 0.143, 0.167, 'hf1_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][2], outputs[1][2], outputs[2], 0.143, 0.167, 'hf2_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][3], outputs[1][3], outputs[2], 0.143, 0.167, 'hf3_'))
            loss_dict.update(reid_losses(self._cfg, outputs[0][4], outputs[1][4], outputs[2], 0.143, 0.167, 'hf_')) 
            loss_dict.update(reid_losses(self._cfg, outputs[0][5], outputs[1][5], outputs[2], 0.143, 0.167, 'pred_')) 
            loss_dict.update(reid_losses(self._cfg, outputs[0][6], None, outputs[4], 0.143, 0.167, 'blackid_')) 
        return loss_dict

    def inference(self, images):
        assert not self.training
        features = self.bone(images)  # (bs, 2048, 16, 8)
        global_feat = self.bone_pool(features)
        global_feat = self.heads(global_feat)       

        #HSA branch
        stn = self.stn(images)
        head = stn[0][0]
        grid_list = stn[1][0]
        head_feat = self.hsa(head)
        hf_1 = self.HAN_1(head_feat[:,:,0:8,:])   
        hf1_pool_feat = self.hsa_1_pool(hf_1) 
        hf_2 = self.HAN_2(head_feat[:,:,8:16,:])
        hf2_pool_feat = self.hsa_2_pool(hf_2)
        hf_3 = self.HAN_3(head_feat[:,:,16:24,:])
        hf3_pool_feat = self.hsa_3_pool(hf_3)
        hf = torch.cat((hf1_pool_feat,hf2_pool_feat,hf3_pool_feat),dim=1)
        
        #Adaptive Attention Branch
        gf = global_feat      
        black_logits, b1_pool_feat = self.classifier_black(gf)
        w, black_logits = self.prob(black_logits)
        w1 = torch.exp(w[:,0]) /(torch.exp(w[:,0])+torch.exp(w[:,1]))
        w2 = torch.exp(w[:,1]) /(torch.exp(w[:,0])+torch.exp(w[:,1]))
        w1 = w1.view(-1,1)
        w2 = w2.view(-1,1)        
        pred_feat = torch.cat((w1*gf,w2*hf),dim=1)
        pred_feat= self.final_head(pred_feat) 

        return nn.functional.normalize(pred_feat)
    
    def HAN_1(self,x):
        c_att = self.han_c_att_1(self.han_pool_1(x).view(x.shape[0],-1))
        c_att = F.sigmoid(c_att).view(x.shape[0],-1,1,1)
        feat = x + torch.mul(x,c_att)
        s_att = F.sigmoid(torch.sum(feat,dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat,s_att)
        return han_feat

    def HAN_2(self,x):
        c_att = self.han_c_att_2(self.han_pool_2(x).view(x.shape[0],-1))
        c_att = F.sigmoid(c_att).view(x.shape[0],-1,1,1)
        feat = x + torch.mul(x,c_att)
        s_att = F.sigmoid(torch.sum(feat,dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat,s_att)
        return han_feat
        
    def HAN_3(self,x):
        c_att = self.han_c_att_3(self.han_pool_3(x).view(x.shape[0],-1))
        c_att = F.sigmoid(c_att).view(x.shape[0],-1,1,1)
        feat = x + torch.mul(x,c_att)
        s_att = F.sigmoid(torch.sum(feat,dim=1)).unsqueeze(1)
        han_feat = torch.mul(feat,s_att)
        return han_feat


