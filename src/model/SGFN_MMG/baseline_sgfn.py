import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.model.model_utils.conv1d import FeatureDimMapper
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers, GraphEdgeAttenNetworkLayers_attn
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti

class SGFN(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11, teacher = False):
        super().__init__('SGFN', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN
        
        self.kd = config.KD.kd
        if self.kd:
            self.kd_method = config.KD.method
            self.temperature = config.KD.temperature

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim
        ## point feature size 512 -> 256
        #dim_point_feature = 256 # 512
        dim_point_feature = self.mconfig.point_feature_size # 512
        
        if self.mconfig.USE_SPATIAL:
            dim_point_feature -= dim_f_spatial-3 # ignore centroid
        
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        if self.mconfig.USE_SELF_ATTENTION:
            self.gcn = GraphEdgeAttenNetworkLayers_attn(
                            self.mconfig.point_feature_size,
                            self.mconfig.edge_feature_size,
                            self.mconfig.DIM_ATTEN,
                            self.mconfig.N_LAYERS,
                            self.mconfig.NUM_HEADS,
                            self.mconfig.GCN_AGGR,
                            flow=self.flow,
                            attention=self.mconfig.ATTENTION,
                            use_edge=self.mconfig.USE_GCN_EDGE,
                            DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)
        else:
            self.gcn = GraphEdgeAttenNetworkLayers(
                                self.mconfig.point_feature_size,
                                self.mconfig.edge_feature_size,
                                self.mconfig.DIM_ATTEN,
                                self.mconfig.N_LAYERS,
                                self.mconfig.NUM_HEADS,
                                self.mconfig.GCN_AGGR,
                                flow=self.flow,
                                attention=self.mconfig.ATTENTION,
                                use_edge=self.mconfig.USE_GCN_EDGE,
                                DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)

        self.obj_predictor = PointNetCls(num_obj_class, in_size=self.mconfig.point_feature_size,
                                 batch_norm=with_bn, drop_out=True)
        
        if mconfig.multi_rel_outputs:
            self.rel_predictor = PointNetRelClsMulti(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor = PointNetRelCls(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)

        #self.init_weight()
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.gcn.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)

    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)

        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp],dim=1)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        
        rel_feature = self.rel_encoder(edge_feature)
        
        obj_center = descriptor[:,:3].clone()
        #print(f'obj_feature: {obj_feature.shape}, obj_center: {obj_center.shape}, edge_indices: {edge_indices.shape}, batch_ids: {batch_ids.shape}')

        gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, rel_feature, edge_indices, obj_center, batch_ids)

        #print(f'gcn_obj_feature: {gcn_obj_feature.shape}, gcn_rel_feature: {gcn_rel_feature.shape}')
        rel_cls = self.rel_predictor(gcn_rel_feature)

        obj_logits = self.obj_predictor(gcn_obj_feature)

        if self.kd:
            return obj_logits, rel_cls, gcn_obj_feature, gcn_rel_feature
        else:  
            return obj_logits, rel_cls

    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration += 1

        obj_pred, rel_pred = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        
        # compute loss for obj
        loss_obj = F.cross_entropy(obj_pred, gt_cls)

        # compute loss for rel
        loss_rel = self.compute_rel_loss(rel_pred, gt_rel_cls, ignore_none_rel, weights_rel)

        # Apply lambda scaling
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r, lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        loss = lambda_o * loss_obj + lambda_r * loss_rel
        self.backward(loss)

        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)

        if not with_log:
            return top_k_obj, top_k_rel, loss_rel.detach(), loss_obj.detach(), loss.detach()

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]

        log = [("train/rel_loss", loss_rel.detach().item()),
            ("train/obj_loss", loss_obj.detach().item()),
            ("train/loss", loss.detach().item()),
            ("train/Obj_R1", obj_topk_list[0]),
            ("train/Obj_R5", obj_topk_list[1]),
            ("train/Obj_R10", obj_topk_list[2]),
            ("train/Pred_R1", rel_topk_list[0]),
            ("train/Pred_R3", rel_topk_list[1]),
            ("train/Pred_R5", rel_topk_list[2])]
        return log
           
    def process_val(self, result_print, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices,  
                    scan_id=None, split_id=None, origin_obj_points=None, batch_ids=None, with_log=False, use_triplet=False):
        if self.kd:
            obj_pred, rel_pred, _, _ = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        else:
            obj_pred, rel_pred = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)

        if result_print:
            result_print.cal_gt_and_predict(rel_pred.detach().cpu(), gt_edges, obj_pred.detach().cpu(), gt_cls.cpu(), scan_id, split_id, origin_obj_points, edge_indices.cpu(), batch_ids.cpu())
            if (top_k_rel <= 1).sum() / len(top_k_rel) < 0.7:
                result_print.error_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
            if (top_k_rel <= 1).sum() / len(top_k_rel) > 0.9:
                if (top_k_obj <= 1).sum() / len(top_k_obj) > 0.8:
                    result_print.correct_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
            result_print.all_scan_num += 1
            
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=False, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores


    def kd_process_train(self, teacher, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None, beta=(0,0)):
        self.iteration += 1
        
        obj_pred, rel_pred, gcn_obj_feature, gcn_rel_feature = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        
        with torch.no_grad():
            t_obj_pred, t_rel_pred, t_gcn_obj_feature, t_gcn_rel_feature = teacher(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)


        # compute loss for obj
        loss_obj = F.cross_entropy(obj_pred, gt_cls)
        # compute loss for rel
        loss_rel = self.compute_rel_loss(rel_pred, gt_rel_cls, ignore_none_rel, weights_rel)

        ## teacher
        # KD logit distillation
        if self.kd_method == 'kl':
             kl_loss_obj, kl_loss_rel = self.logit_kl_divergence(obj_pred, t_obj_pred, rel_pred, t_rel_pred)
        
        # KD feature distillation 
        elif self.kd_method == 'feature':
            obj_feature_loss, rel_feature_loss = self.feature_distillation(gcn_obj_feature, t_gcn_obj_feature, gcn_rel_feature, t_gcn_rel_feature)
        
        # KD logit + feature distillation
        elif self.kd_method == 'fkl':
            kl_loss_obj, kl_loss_rel = self.logit_kl_divergence(obj_pred, t_obj_pred, rel_pred, t_rel_pred)
            obj_feature_loss, rel_feature_loss = self.feature_distillation(gcn_obj_feature, t_gcn_obj_feature, gcn_rel_feature, t_gcn_rel_feature)
        elif self.kd_method == 'fmse':
            t_mse_loss_obj = F.mse_loss(obj_pred, t_obj_pred)
            t_bce_loss_rel = self.compute_rel_loss(rel_pred, t_rel_pred, ignore_none_rel, weights_rel)
            obj_feature_loss, rel_feature_loss = self.feature_distillation(gcn_obj_feature, t_gcn_obj_feature, gcn_rel_feature, t_gcn_rel_feature)
        # KD mse distillation
        else:
            t_mse_loss_obj = F.mse_loss(obj_pred, t_obj_pred)
            t_bce_loss_rel = self.compute_rel_loss(rel_pred, t_rel_pred, ignore_none_rel, weights_rel)
       
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        beta_o = beta[0]
        alpha_o = 1- beta_o
        beta_r = beta[1]
        alpha_r = 1- beta_r

        if self.kd_method == 'kl':
            t_loss_obj, t_loss_rel = kl_loss_obj, kl_loss_rel
        elif self.kd_method == 'feature':
            t_loss_obj, t_loss_rel = obj_feature_loss, rel_feature_loss
        elif self.kd_method == 'fkl':
            t_loss_obj, t_loss_rel = kl_loss_obj + obj_feature_loss, kl_loss_rel + rel_feature_loss
        elif self.kd_method == 'fmse':
            t_loss_obj, t_loss_rel = t_mse_loss_obj + obj_feature_loss, t_bce_loss_rel + rel_feature_loss
        elif self.kd_method == 'mse':
            t_loss_obj, t_loss_rel = t_mse_loss_obj, t_bce_loss_rel
        loss = lambda_o * (alpha_o * loss_obj + beta_o * t_loss_obj) + lambda_r * (alpha_r * loss_rel + beta_r * t_loss_rel)
        
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if not with_log:
            return top_k_obj, top_k_rel, loss_rel.detach(), loss_obj.detach(), loss.detach()

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/t_loss_obj", t_loss_obj.detach().item()),
                ("train/t_rel_loss", t_loss_rel.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        if self.kd_method == 'fkl':
            log.insert(8, ("train/T_kl_loss_rel", kl_loss_rel.detach().item()))
            log.insert(9, ("train/T_kl_loss_obj", kl_loss_obj.detach().item()))
        if self.kd_method == 'fmse' or self.kd_method == 'fkl':
            log.insert(8, ("train/T_rel_feature_loss", rel_feature_loss if isinstance(rel_feature_loss, float) else rel_feature_loss.detach().item()))
            log.insert(9, ("train/T_obj_feature_loss", obj_feature_loss if isinstance(obj_feature_loss, float) else obj_feature_loss.detach().item()))
        if self.kd_method == 'fmse':
            log.insert(8, ("train/T_bce_loss_rel", t_bce_loss_rel.detach().item()))
            log.insert(9, ("train/T_mse_loss_obj", t_mse_loss_obj.detach().item()))
        return log
    
    def compute_weights(self, gt_rel_cls, ignore_none_rel, weights_rel):
        """ Compute the weight for the loss function based on the configuration. """
        if self.mconfig.WEIGHT_EDGE == 'BG':
            if self.mconfig.w_bg != 0:
                return self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
            else:
                return None
        elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
            batch_mean = torch.sum(gt_rel_cls, dim=(0))
            zeros = (gt_rel_cls.sum(-1) == 0).sum().unsqueeze(0)
            batch_mean = torch.cat([zeros, batch_mean], dim=0)
            weight = torch.abs(1.0 / (torch.log(batch_mean + 1) + 1))  # +1 to prevent 1 /log(1) = inf
            if ignore_none_rel:
                weight[0] = 0
                weight *= 1e-2  # reduce the weight from ScanNet
            if 'NONE_RATIO' in self.mconfig:
                weight[0] *= self.mconfig.NONE_RATIO

            weight[torch.where(weight == 0)] = weight[0].clone() if not ignore_none_rel else 0  # * 1e-3
            return weight[1:]
        elif self.mconfig.WEIGHT_EDGE == 'OCCU':
            return weights_rel
        elif self.mconfig.WEIGHT_EDGE == 'NONE':
            return None
        else:
            raise NotImplementedError("unknown weight_edge type")
    
    def compute_rel_loss(self, rel_pred, gt_rel_cls, ignore_none_rel, weights_rel):
        """ Compute the relationship loss based on configuration. """
        if self.mconfig.multi_rel_outputs:
            weight = self.compute_weights(gt_rel_cls, ignore_none_rel, weights_rel)
            return F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight)
        else:
            weight = self.compute_weights(gt_rel_cls, ignore_none_rel, weights_rel)
            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                return torch.zeros(1, device=rel_pred.device, requires_grad=False)
            else:
                return F.nll_loss(rel_pred, gt_rel_cls, weight=weight)
            

    def logit_kl_divergence(self, obj_pred, obj_target, rel_pred, rel_target):
        T = self.temperature

        # 온도 스케일링 적용
        obj_pred_scaled = obj_pred / T
        obj_target_scaled = obj_target / T
        rel_pred_scaled = rel_pred / T
        rel_target_scaled = rel_target / T

        # KL Divergence Loss
        kl_loss_obj = F.kl_div(
            F.log_softmax(obj_pred_scaled, dim=1),
            F.softmax(obj_target_scaled, dim=1),
            reduction='batchmean'
        ) * (T * T)
        kl_loss_rel = F.kl_div(
            F.log_softmax(rel_pred_scaled, dim=1),
            F.softmax(rel_target_scaled, dim=1),
            reduction='batchmean'
        ) * (T * T)

        return kl_loss_obj, kl_loss_rel
    
    def feature_distillation(self, obj_feature, obj_feature_target, rel_feature, rel_feature_target):
        obj_feature_distillation_loss, rel_feature_distillation_loss =0.0,0.0
        if self.reduced_point_dim:
            obj_feature_target = self.obj_feature_dim_mapper(obj_feature_target)
            obj_feature_distillation_loss = F.mse_loss(obj_feature, obj_feature_target)

        if self.reduced_edge_dim:
            rel_feature_target = self.edge_feature_dim_mapper(rel_feature_target)
            rel_feature_distillation_loss = F.mse_loss(rel_feature, rel_feature_target)
        return obj_feature_distillation_loss, rel_feature_distillation_loss
    
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()