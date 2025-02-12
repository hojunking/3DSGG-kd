import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers
from src.model.model_utils.network_TripletGCN import TripletGCNModel
from src.model.model_utils.conv1d import FeatureDimMapper
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti

class SGPN(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11, teacher = False):
        super().__init__('SGPN', config)

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
        
        dim_point = 3
        dim_point_rel = 3
        if mconfig.USE_RGB:
            dim_point +=3
            dim_point_rel+=3
        if mconfig.USE_NORMAL:
            dim_point +=3
            dim_point_rel+=3
            
        if mconfig.USE_CONTEXT:
            dim_point_rel += 1

        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'

        dim_point_feature = 256

        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.point_feature_size)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        self.gcn = TripletGCNModel(
            num_layers=self.mconfig.N_LAYERS,
            dim_node=mconfig.point_feature_size,
            dim_edge=mconfig.edge_feature_size,
            dim_hidden=mconfig.DIM_ATTEN,
            use_bn=with_bn)

        self.obj_predictor = PointNetCls(num_obj_class, in_size=mconfig.point_feature_size,
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
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            #{'params':self.mlp.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()


    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)


        rel_feature = self.rel_encoder(edge_feature)

        #print(f'obj_feature: {obj_feature.shape}, rel_feature: {rel_feature.shape}, edge_indices: {edge_indices.shape}')
        gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edge_indices)

        rel_cls = self.rel_predictor(gcn_rel_feature)

        obj_logits = self.obj_predictor(gcn_obj_feature)

        if self.kd:
            return obj_logits, rel_cls, gcn_obj_feature, gcn_rel_feature
        else:  
            return obj_logits, rel_cls
        
    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        
        obj_pred, rel_pred = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(),descriptor, batch_ids, istrain=True)
        
        # compute loss for obj
        loss_obj = F.nll_loss(obj_pred, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
        else:
            loss_rel = F.nll_loss(rel_pred, gt_rel_cls)

        
        loss = 0.1 * loss_obj + loss_rel
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

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
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
    
    
    def kd_process_train(self, teacher, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None, beta =0.1):
        self.iteration +=1    
        
        
        obj_pred, rel_pred, gcn_obj_feature, gcn_rel_feature = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        
        with torch.no_grad():
            t_obj_pred, t_rel_pred, t_gcn_obj_feature, t_gcn_rel_feature = teacher(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)

        # compute loss for obj
        loss_obj = F.nll_loss(obj_pred, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
        else:
            loss_rel = F.nll_loss(rel_pred, gt_rel_cls)

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
            t_bce_loss_rel = F.binary_cross_entropy(rel_pred, t_rel_pred)
            obj_feature_loss, rel_feature_loss = self.feature_distillation(gcn_obj_feature, t_gcn_obj_feature, gcn_rel_feature, t_gcn_rel_feature)

        # KD mse distillation
        else:
            t_mse_loss_obj = F.mse_loss(obj_pred, t_obj_pred)
            t_bce_loss_rel = F.binary_cross_entropy(rel_pred, t_rel_pred)

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

        loss = 0.1 * (alpha_o * loss_obj + beta_o * t_loss_obj) + (alpha_r * loss_rel) #+ beta_r * t_loss_rel)
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/t_loss_obj", t_loss_obj.detach().item()),
                ("train/t_rel_loss", t_loss_rel.detach().item()),
                ("train/loss", loss.detach().item()),
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