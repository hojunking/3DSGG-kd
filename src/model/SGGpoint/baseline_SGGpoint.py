import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
from src.model.model_utils.conv1d import FeatureDimMapper
from model.model_utils.model_base import BaseModel
from src.model.model_utils.network_PointNet import PointNetfeat
from src.utils.eva_utils_acc import ( evaluate_topk_object,
                                 evaluate_topk_predicate,
                                 evaluate_triplet_topk, get_gt)
from utils import op_utils


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNN(nn.Module):
    # official DGCNN
    def __init__(self, input_channel, embeddings):
        super(DGCNN, self).__init__()
        self.k = 20
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, embeddings, kernel_size=1, bias=False),nn.BatchNorm1d(embeddings),nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv5(x)
        return x

##################################################
#                                                #
#                                                #
#  Core Network: EdgeGCN                         #
#                                                #
#                                                #
##################################################

class EdgeGCN(torch.nn.Module):
    def __init__(self, num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag):
        super(EdgeGCN, self).__init__()

        self.node_GConv1 = GCNConv(num_node_in_embeddings, num_node_in_embeddings // 2, add_self_loops=True)
        self.node_GConv2 = GCNConv(num_node_in_embeddings // 2, num_node_in_embeddings, add_self_loops=True)

        self.edge_MLP1 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings, num_edge_in_embeddings // 2, 1), nn.ReLU())
        self.edge_MLP2 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings // 2, num_edge_in_embeddings, 1), nn.ReLU())

        self.AttnEdgeFlag = AttnEdgeFlag # boolean (for ablaiton studies)
        self.AttnNodeFlag = AttnNodeFlag # boolean (for ablaiton studies)

        # multi-dimentional (N-Dim) node/edge attn coefficients mappings
        self.edge_attentionND = nn.Linear(num_edge_in_embeddings, num_node_in_embeddings // 2) if self.AttnEdgeFlag else None
        self.node_attentionND = nn.Linear(num_node_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

        self.node_indicator_reduction = nn.Linear(num_edge_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None
        
        self.obj_mlp = nn.Linear(num_node_in_embeddings*2, num_node_in_embeddings)
        self.rel_mlp = nn.Linear(num_edge_in_embeddings*2, num_edge_in_embeddings)

    def concate_NodeIndicator_for_edges(self, node_indicator, batchwise_edge_index):
        node_indicator = node_indicator.squeeze(0)
        
        edge_index_list = batchwise_edge_index.t()
        subject_idx_list = edge_index_list[:, 0]
        object_idx_list = edge_index_list[:, 1]

        subject_indicator = node_indicator[subject_idx_list]  # (num_edges, num_mid_channels)
        object_indicator = node_indicator[object_idx_list]    # (num_edges, num_mid_channels)

        edge_concat = torch.cat((subject_indicator, object_indicator), dim=-1)
        return edge_concat  # (num_edges, num_mid_channels * 2)

    def forward(self, node_feats, edge_feats, edge_index):
        # prepare node_feats & edge_feats in the following formats
        # node_feats: (1, num_nodes,  num_embeddings)
        # edge_feats: (1, num_edges,  num_embeddings)
        # (num_embeddings = num_node_in_embeddings = num_edge_in_embeddings) = 2 * num_mid_channels
        node_feats_ = node_feats.clone()
        edge_feats_ = edge_feats.clone()

        #### Deriving Edge Attention
        if self.AttnEdgeFlag:
            edge_indicator = self.edge_attentionND(edge_feats.squeeze(0)).unsqueeze(0).permute(0, 2, 1)  # (1, num_mid_channels, num_edges)
            raw_out_row = scatter(edge_indicator, edge_index.t()[:, 0].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            raw_out_col = scatter(edge_indicator, edge_index.t()[:, 1].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            agg_edge_indicator_logits = raw_out_row * raw_out_col                                        # (1, num_mid_channels, num_nodes)
            agg_edge_indicator = torch.sigmoid(agg_edge_indicator_logits).permute(0, 2, 1).squeeze(0)    # (num_nodes, num_mid_channels)
        else:
            agg_edge_indicator = 1

        #print(f'node_feats: {node_feats.shape}, edge_feats: {edge_feats.shape}, edge_index: {edge_index.shape}')
        #### Node Evolution Stream (NodeGCN)
        node_feats = F.relu(self.node_GConv1(node_feats, edge_index)) * agg_edge_indicator # applying EdgeAttn on Nodes
        node_feats = F.dropout(node_feats, training=self.training)
        node_feats = F.relu(self.node_GConv2(node_feats, edge_index))
        node_feats = node_feats.unsqueeze(0)  # (1, num_nodes, num_embeddings)

        #### Deriving Node Attention
        if self.AttnNodeFlag:
            node_indicator = F.relu(self.node_attentionND(node_feats.squeeze(0)).unsqueeze(0))                  # (1, num_mid_channels, num_nodes)
            agg_node_indicator = self.concate_NodeIndicator_for_edges(node_indicator, edge_index)               # (num_edges, num_mid_channels * 2)
            agg_node_indicator = self.node_indicator_reduction(agg_node_indicator).unsqueeze(0).permute(0,2,1)  # (1, num_mid_channels, num_edges)
            agg_node_indicator = torch.sigmoid(agg_node_indicator)  # (1, num_mid_channels, num_edges)
        else:
            agg_node_indicator = 1

        #### Edge Evolution Stream (EdgeMLP)
        edge_feats = edge_feats.unsqueeze(0).permute(0, 2, 1)                  # (1, num_embeddings, num_edges)
        edge_feats = self.edge_MLP1(edge_feats)                   # (1, num_mid_channels, num_edges)
        edge_feats = F.dropout(edge_feats, training=self.training) * agg_node_indicator    # applying NodeAttn on Edges
        edge_feats = self.edge_MLP2(edge_feats).permute(0, 2, 1)  # (1, num_edges, num_embeddings)

        node_feats = node_feats.squeeze(0)
        edge_feats = edge_feats.squeeze(0)
        node_feats = self.obj_mlp(torch.cat([node_feats_, node_feats.squeeze(0)], dim=-1))
        edge_feats = self.rel_mlp(torch.cat([edge_feats_, edge_feats.squeeze(0)], dim=-1))

        return  node_feats, edge_feats

###############################################
#                                             #
#                                             #
#   Tail Classification - NodeMLP & EdgeMLP   #
#                                             #
#                                             #
###############################################

class NodeMLP(nn.Module):
    def __init__(self, embeddings, nObjClasses, negative_slope=0.2):
        super(NodeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.node_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.node_BnReluDp = nn.Sequential(nn.BatchNorm1d(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.node_linear2 = nn.Linear(mid_channels, nObjClasses, bias=False)

    def forward(self, node_feats):
        # node_feats: (1, nodes, embeddings)  => node_logits: (1, nodes, nObjClasses)
        x = self.node_linear1(node_feats.unsqueeze(0))
        x = self.node_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        node_logits = self.node_linear2(x)
        return node_logits.squeeze(0)

class EdgeMLP(nn.Module):
    def __init__(self, embeddings, nRelClasses, negative_slope=0.2):
        super(EdgeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.edge_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(nn.BatchNorm1d(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.edge_linear2 = nn.Linear(mid_channels, nRelClasses, bias=False)

    def forward(self, edge_feats):
        # edge_feats: (1, edges, embeddings)  => edge_logits: (1, edges, nRelClasses)
        x = self.edge_linear1(edge_feats.unsqueeze(0))
        x = self.edge_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_linear2(x)
        # we treat it as multi-label classification
        edge_logits = torch.sigmoid(edge_logits)
        return edge_logits.squeeze(0)

#####################################################
#                                                   #
#                                                   #
#   SGGpoint Model                                  #
#                                                   #
#                                                   #
#####################################################

def edge_feats_initialization(node_feats, batchwise_edge_index):

    connections_from_subject_to_object = batchwise_edge_index.t()
    subject_idx = connections_from_subject_to_object[:, 0]
    object_idx = connections_from_subject_to_object[:, 1]

    subject_feats = node_feats[subject_idx]
    object_feats = node_feats[object_idx]
    diff_feats = object_feats - subject_feats

    edge_feats = torch.cat((subject_feats, diff_feats), dim=1)  # equivalent to EdgeConv (with in DGCNN)

    return edge_feats  # (num_Edges, Embeddings * 2)

class SGGpoint(BaseModel):
    # architecture
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11, teacher = False):
        super().__init__('SGGpoint', config)

        self.config = config
        self.mconfig = mconfig = config.MODEL

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
        
        dim_point_feature = self.mconfig.point_feature_size # 512

        if self.mconfig.USE_SPATIAL:
            dim_point_feature -= dim_f_spatial-3
        # self.backbone = nn.Sequential(
        #     DGCNN(input_channel=3, embeddings=self.mconfig.point_feature_size -8)
        # )

        # Object Encoder
        self.backbone = PointNetfeat(
            global_feat=True, 
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)
        
       # self.mlp_3d = torch.nn.Linear(512, self.mconfig.point_feature_size -8)
        self.edge_mlp = nn.Linear(self.mconfig.point_feature_size*2, self.mconfig.edge_feature_size - 11)
        self.gcn = EdgeGCN(num_node_in_embeddings=self.mconfig.point_feature_size,
                                num_edge_in_embeddings=self.mconfig.edge_feature_size,
                                AttnNodeFlag=True, AttnEdgeFlag=True)
        self.obj_classifier = NodeMLP(embeddings=self.mconfig.point_feature_size, nObjClasses=num_obj_class)
        self.rel_classifier = EdgeMLP(embeddings=self.mconfig.edge_feature_size, nRelClasses=num_rel_class)
        
        if self.kd and teacher != True:
            self.reduced_point_dim = False
            self.reduced_edge_dim = False
            if '_p_' in config.exp:
                self.reduced_point_dim = True
                
                self.obj_feature_dim_mapper = FeatureDimMapper(
                    self.mconfig.point_feature_size*2,
                    self.mconfig.point_feature_size)
            if '_e_' in config.exp:
                self.reduced_edge_dim = True  
                self.edge_feature_dim_mapper = FeatureDimMapper(
                    self.mconfig.edge_feature_size*2,
                    self.mconfig.edge_feature_size)
        
        self.obj_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.optimizer = optim.Adam([
            {'params':self.backbone.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.gcn.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.edge_mlp.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            # {'params':self.obj_mlp.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            # {'params':self.rel_mlp.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.obj_classifier.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.rel_classifier.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.obj_logit_scale, 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()
        
        
    def init_weight(self, obj_label_path, rel_label_path, adapter_path):
        # torch.nn.init.xavier_uniform_(self.obj_mlp.weight)
        # torch.nn.init.xavier_uniform_(self.rel_mlp.weight)
        torch.nn.init.xavier_uniform_(self.edge_mlp.weight)
        
        self.obj_logit_scale.requires_grad = True
        
        
    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        # Generate node initial feature
        node_feats = self.backbone(obj_points)
        #node_feats = torch.max(x, 2)[0] # perform maxpooling

        #add
        #node_feats = self.mlp_3d(node_feats)
        #print(f'node_feats: {node_feats.shape}')
        # Generate edge initial feature
        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            node_feats = torch.cat([node_feats, tmp],dim=-1)
        
        edge_feats = edge_feats_initialization(node_feats.clone(), edge_indices)
        #print(f'node_feats: {node_feats.shape}, edge_feats: {edge_feats.shape}, edge_indices: {edge_indices.shape}')
        edge_feats = self.edge_mlp(edge_feats)
        with torch.no_grad():
            x_i = descriptor[edge_indices[0]]
            x_j = descriptor[edge_indices[1]]
            edge_feats_des = torch.zeros_like(x_i)
            edge_feats_des[:,0:3] = x_i[:,0:3]-x_j[:,0:3]
            # std  offset
            edge_feats_des[:,3:6] = x_i[:,3:6]-x_j[:,3:6]
            # dim log ratio
            edge_feats_des[:,6:9] = torch.log(x_i[:,6:9] / x_j[:,6:9])
            # volume log ratio
            edge_feats_des[:,9] = torch.log( x_i[:,9] / x_j[:,9])
            # length log ratio
            edge_feats_des[:,10] = torch.log( x_i[:,10] / x_j[:,10])

        edge_feats = torch.cat((edge_feats, edge_feats_des), dim=-1)
        
        # EdgeGCN
        node_feats_gcn, edge_feats_gcn = self.gcn(node_feats, edge_feats, edge_indices)
        
        logit_scale = self.obj_logit_scale.exp()
        obj_logits = logit_scale * self.obj_classifier(node_feats_gcn / node_feats_gcn.norm(dim=-1, keepdim=True))
        
        #obj_logits = self.obj_classifier(node_feats)
        rel_logits = self.rel_classifier(edge_feats_gcn.squeeze(0))
        
        if self.kd:
            return obj_logits, rel_logits, logit_scale, node_feats_gcn, edge_feats_gcn
        else:
            return obj_logits, rel_logits, logit_scale
    
    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration += 1 

        obj_logits_3d, rel_cls_3d, logit_scale = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        loss_obj_3d = F.cross_entropy(obj_logits_3d, gt_cls)
        
        batch_mean = torch.sum(gt_rel_cls, dim=(0))
        zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
        batch_mean = torch.cat([zeros,batch_mean],dim=0)
        weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
            
        weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
        weight = weight[1:]
        loss_rel_3d = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls, weight=weight)
        
        loss = 0.1 * loss_obj_3d + 3 * loss_rel_3d
        self.backward(loss)

        top_k_obj = evaluate_topk_object(obj_logits_3d.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]        
        
        log = [("train/rel_loss", loss_rel_3d.detach().item()),
                ("train/obj_loss", loss_obj_3d.detach().item()),
                ("train/logit_scale", logit_scale.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
    
    def kd_process_train(self, teacher, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None, beta=(0,0)):
        self.iteration += 1 
        obj_pred, rel_pred, logit_scale, gcn_obj_feature, gcn_rel_feature  = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        with torch.no_grad():
            t_obj_pred, t_rel_pred, _, t_gcn_obj_feature, t_gcn_rel_feature = teacher(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        loss_obj_3d = F.cross_entropy(obj_pred, gt_cls)
        
        batch_mean = torch.sum(gt_rel_cls, dim=(0))
        zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
        batch_mean = torch.cat([zeros,batch_mean],dim=0)
        weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
            
        weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
        weight = weight[1:]    
        loss_rel_3d = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight)

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
            t_bce_loss_rel =  F.binary_cross_entropy(rel_pred, t_rel_pred, weight=weight)
            obj_feature_loss, rel_feature_loss = self.feature_distillation(gcn_obj_feature, t_gcn_obj_feature, gcn_rel_feature, t_gcn_rel_feature)
        # KD mse distillation
        else:
            t_mse_loss_obj = F.mse_loss(obj_pred, t_obj_pred)
            t_bce_loss_rel =  F.binary_cross_entropy(rel_pred, t_rel_pred, weight=weight)

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
        
        loss = 0.1 * (alpha_o* loss_obj_3d + beta_o * t_loss_obj) + 3 * (alpha_r * loss_rel_3d + beta_r*t_loss_rel)

        self.backward(loss)

        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]        
        
        log = [("train/rel_loss", loss_rel_3d.detach().item()),
                ("train/obj_loss", loss_obj_3d.detach().item()),
                ("train/logit_scale", logit_scale.detach().item()),
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

    def process_val(self, result_print, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices,  
                    scan_id=None, split_id=None, origin_obj_points=None, batch_ids=None, with_log=False, use_triplet=False):
        if self.kd:
            obj_logits_3d, rel_cls_3d, _, _, _ = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        else:
            obj_logits_3d, rel_cls_3d, _ = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if result_print:
            result_print.cal_gt_and_predict(rel_cls_3d.detach().cpu(), gt_edges, obj_logits_3d.detach().cpu(), gt_cls.cpu(), scan_id, split_id, origin_obj_points, edge_indices.cpu(), batch_ids.cpu())
            if (top_k_rel <= 1).sum() / len(top_k_rel) < 0.7:
                result_print.error_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
            if (top_k_rel <= 1).sum() / len(top_k_rel) > 0.9:
                if (top_k_obj <= 1).sum() / len(top_k_obj) > 0.8:
                    result_print.correct_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
            result_print.all_scan_num += 1
        
        top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
        
        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
    
    def val_loss(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration += 1 

        obj_logits_3d, rel_cls_3d, _ = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        loss_obj_3d = F.cross_entropy(obj_logits_3d, gt_cls)
        
        batch_mean = torch.sum(gt_rel_cls, dim=(0))
        zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
        batch_mean = torch.cat([zeros,batch_mean],dim=0)
        weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
            
        weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
        weight = weight[1:]
        loss_rel_3d = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls, weight=weight)

        loss = 0.1 * loss_obj_3d + 3 * loss_rel_3d
        self.backward2(loss)

        return loss
    

    def logit_kl_divergence(self, obj_pred, obj_target, rel_pred, rel_target):
        T = self.temperature

        # 온도 스케일링 적용
        obj_pred_scaled = self.logit_scaling(obj_pred)
        obj_target_scaled = self.logit_scaling(obj_target)
        rel_pred_scaled = self.logit_scaling(rel_pred)
        rel_target_scaled = self.logit_scaling(rel_target)

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


    def logit_scaling(self, logits):
        T = self.temperature
        return logits / T
    

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()

    def backward2(self, loss):
        self.zero_grad()
        loss.backward()