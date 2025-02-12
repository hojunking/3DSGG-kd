if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import copy
import os, glob, time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg)
from src.dataset.dataset_builder import build_dataset
from src.model.SGFN_MMG.model import Mmgnet
from src.model.SGFN_MMG.baseline_sgfn import SGFN
from src.model.SGFN_MMG.baseline_sgpn import SGPN
from src.model.SGGpoint.baseline_SGGpoint import SGGpoint
from src.model.SGFN_MMG.baseline_imp import IMP
from src.utils import op_utils
from src.utils.eva_utils_acc import get_mean_recall, get_zero_shot_recall
import torch_geometric
from utils.gspread import save_gspread
from utils.estimators import mlp_importance_estimator, distillation_rate_estimator
from visualization.taylor_visualization import visualize_taylor_scores
from process_data.relation_distribution import Result_print
# pruning
import torch_pruning as tp
from functools import partial
from fvcore.nn import FlopCountAnalysis
import torch.nn.utils.prune as prune

class MMGNet():
    def __init__(self, config):
        self.config = config
        self.previous_ratio = 0
        self.scores = None
        self.model_name = self.config.NAME
        self.mconfig = mconfig = config.MODEL
        self.exp = config.exp
        self.save_res = config.EVAL
        self.update_2d = config.update_2d
        self.masks = {}
        self.start_time, self.end_time = 0, 0
        self.st_pruning_ratio = config.pruning.st_pruning_ratio
        self.unst_pruning_ratio = config.pruning.unst_pruning_ratio
        self.obj_ratio, self.rel_ratio = 0, 0
        self.baseline_acc, self.phase1_acc = self.config.pruning.baseline_result, self.config.pruning.phase1_result
        
        

        ''' pruning config '''
        self.prune_method = config.pruning.method
        self.prune_reg = config.pruning.reg
        self.prune_delta_reg = config.pruning.delta_reg
        self.prune_global_pruning = config.pruning.global_pruning
        
        # real pruning ratio
        self.encoder_pruned_ratio = 0
        self.gnn_pruned_ratio = 0
        self.classifier_pruned_ratio = 0
        ''' Build dataset '''
        if config.MODE  == 'train' or config.MODE == 'prune':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config,split_type='train_scans', shuffle_objs=True,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            self.dataset_train.__getitem__(0)

        if self.config.pruning_method in ['taylor_co1', 'taylor_co2','taylor_point', 'taylor_edge']:
            self.dataset_train = build_dataset(self.config,split_type='train_scans', shuffle_objs=False,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            self.dataset_train.__getitem__(0)
        #elif config.MODE  == 'train' or config.MODE  == 'trace' or config.MODE  == 'eval' or config.MODE == 'prune':
        if config.VERBOSE: print('build valid dataset')
        self.dataset_valid = build_dataset(self.config,split_type='validation_scans', shuffle_objs=False, 
                                    multi_rel_outputs=mconfig.multi_rel_outputs,
                                    use_rgb=mconfig.USE_RGB,
                                    use_normal=mconfig.USE_NORMAL)

        num_obj_class = len(self.dataset_valid.classNames)   
        num_rel_class = len(self.dataset_valid.relationNames)
        self.num_obj_class = num_obj_class
        self.num_rel_class = num_rel_class
        
        if config.MODE  in ['train', 'prune']:
            self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_train) // self.config.Batch_Size)
        
        elif config.MODE  == 'eval':
            if self.config.pruning_method in ['taylor_co1', 'taylor_co2', 'taylor_point', 'taylor_edge']:
                self.total = self.config.total = 548 // self.config.Batch_Size
                self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*548 // self.config.Batch_Size)
                self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*548 // self.config.Batch_Size)
            else:
                self.total = self.config.total = len(self.dataset_valid) // self.config.Batch_Size
                self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_valid) // self.config.Batch_Size)
                self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_valid) // self.config.Batch_Size)

        # 모델 클래스를 딕셔너리에 매핑
        model_classes = {
            'Mmgnet': Mmgnet,
            'sgfn': SGFN,
            'sgfnattn': SGFN,
            'sgpn': SGPN,
            'SGGpoint': SGGpoint,
            'imp':IMP
        }

        print(f'model name : {self.model_name}')
        # 모델 이름이 올바른지 확인
        if self.model_name in model_classes:
            model_class = model_classes[self.model_name]
            self.model = model_class(self.config, num_obj_class, num_rel_class).to(config.DEVICE)
            
        else:
            print(f'Unknown model name: {self.model_name}')
            raise NotImplementedError
        
        ## load pre-trained weights
        if self.mconfig.use_pretrain != "":
            if self.config.pruning.load_pruning_model:
                ## load pre-trained pruned weights
                self.model = torch.load(os.path.join(self.mconfig.use_pretrain))
            elif self.config.MODE == 'prune':
                ## load pre-trained weights and pruning 
                print(f'load pretrain model: {self.mconfig.use_pretrain}')
                self.model.load_pretrain_model(self.mconfig.use_pretrain, skip_names=['obj_feature_dim_mapper', 'edge_feature_dim_mapper'], is_freeze=False)
                if self.config.KD.kd != True:
                    self.config.pruning.load_pruning_model = True
            elif self.config.MODE == 'eval':
                ## load pre-trained weights and pruning 
                print(f'load pretrain model: {self.mconfig.use_pretrain}')
                self.model.load_pretrain_model(self.mconfig.use_pretrain, skip_names=['obj_feature_dim_mapper', 'edge_feature_dim_mapper'], is_freeze=False)

        if self.config.KD.kd and self.config.KD.t_model_path != "":
            self.recover_submodules()
            self.t_model = model_class(self.config, num_obj_class, num_rel_class, teacher = True).to(config.DEVICE)

            
            print(f'load teacher model: {self.config.KD.t_model_path}')
            self.t_model.load_pretrain_model(self.config.KD.t_model_path, is_freeze=True)
            self.config.pruning.load_pruning_model = True

            self.beta = distillation_rate_estimator(self.baseline_acc[0], self.baseline_acc[1], self.phase1_acc[0], self.phase1_acc[1], self.config.KD.gamma)
            print(f'KD obj_beta: {self.beta[0]}, rel_beta: {self.beta[1]}')
            #else:
            ### pruning + KD code 필요 (2 pre-trained weight)

        self.samples_path = os.path.join(config.PATH, self.model_name, self.exp,  'samples')
        self.results_path = os.path.join(config.PATH, self.model_name, self.exp, 'results')
        self.trace_path = os.path.join(config.PATH, self.model_name, self.exp, 'traced')
        self.writter = None
        
        if not self.config.EVAL:
            pth_log = os.path.join(config.PATH, "logs", self.model_name, self.exp)
            self.writter = SummaryWriter(pth_log)
    #def gspread_task(self):
    def load(self, best=False):
        return self.model.load(best)

    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, _, _, _ = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
    
    @torch.no_grad()
    def data_processing_val(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id, split_id, origin_obj_points = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id, split_id, origin_obj_points

    def get_pruner(self, model, example_inputs, num_classes, ignored_layers=[]):
        self.config.pruning.sparsity_learning = False
        if self.prune_method == "random":
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "l1":
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "l2":
            imp = tp.importance.MagnitudeImportance(p=2)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "fpgm":
            imp = tp.importance.FPGMImportance(p=2)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "obdc":
            imp = tp.importance.OBDCImportance(group_reduction='mean', num_classes=self.num_obj_class)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "lamp":
            imp = tp.importance.LAMPImportance(p=2)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "slim":
            self.config.pruning.sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=self.prune_reg, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "group_slim":
            self.config.pruning.sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=self.prune_reg, global_pruning=self.prune_global_pruning, group_lasso=True)
        elif self.prune_method == "group_norm":
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "group_sl":
            self.config.pruning.sparsity_learning = True
            imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
            pruner_entry = partial(tp.pruner.GroupNormPruner, reg=self.prune_reg, global_pruning=self.prune_global_pruning)
        elif self.prune_method == "growing_reg":
            self.config.pruning.sparsity_learning = True
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=self.prune_reg, delta_reg=self.prune_delta_reg, global_pruning=self.config.global_pruning)
        else:
            raise NotImplementedError
        unwrapped_parameters = []
        pruning_ratio_dict = {}

        # ignore output layers
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features in num_classes:
                ignored_layers.append(m)
            elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels in num_classes:
                ignored_layers.append(m)
        
        # Here we fix iterative_steps=200 to prune the model progressively with small steps 
        # until the required speed up is achieved.
        #print("ignored layers: ", ignored_layers)
        pruner = pruner_entry(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=self.config.pruning.iterative_steps,
            pruning_ratio=1.0,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=self.config.pruning.max_pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )
        return pruner

    def train(self):
        print('===   start training   ===')
        self.start_time = time.time()
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=self.config.WORKERS,
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg,
        )

        self.model.epoch = 1
        keep_training = True
        
        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it', 'Misc/lr'])
                
        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)

        ''' Train '''
        while(keep_training):

            if self.model.epoch > self.config.MAX_EPOCHES:
                break
            print('\n\nTraining epoch: %d' % self.model.epoch)
    
            for items in loader:

                self.model.train()
                ''' get data '''
                obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(items)
                ### KD Training ###
                if self.config.KD.kd:
                    self.t_model.eval()
                    logs = self.model.kd_process_train(self.t_model, obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, with_log=True,
                                                weights_obj=self.dataset_train.w_cls_obj,
                                                weights_rel=self.dataset_train.w_cls_rel,
                                                ignore_none_rel = False,
                                                beta=self.beta)
                else:
                    logs = self.model.process_train(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, with_log=True,
                                                    weights_obj=self.dataset_train.w_cls_obj, 
                                                    weights_rel=self.dataset_train.w_cls_rel,
                                                    ignore_none_rel = False)
                iteration = self.model.iteration
                logs += [
                    ("Misc/epo", int(self.model.epoch)),
                    ("Misc/it", int(iteration)),
                    ("lr", self.model.lr_scheduler.get_last_lr()[0])
                ]
                
                progbar.add(1, values=logs \
                            if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs, iteration)
                if self.model.iteration >= self.max_iteration:
                    break
            progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            loader = iter(train_loader)
            self.save()

            if (self.model.epoch > 20 and 'VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and self.model.epoch % self.config.VALID_INTERVAL == 0):
                print('start validation...')
                rel_acc_val = self.validation()
                self.model.eva_res = rel_acc_val
                self.save()
            
            #self.track_pruned_weights()
            self.model.epoch += 1
            if self.model.epoch > 100 : 
                self.config.VALID_INTERVAL = 10 
                   
    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]
    
    def log(self, logs, iteration):
        # Tensorboard
        if self.writter is not None and not self.config.EVAL:
            for i in logs:
                if not i[0].startswith('Misc'):
                    self.writter.add_scalar(i[0], i[1], iteration)
                    
    def save(self):
        self.model.save ()

    def get_sample_loader(self):
        import pickle
        import random

        index_file = 'data_processing/sample_idx.pkl'
        if os.path.exists(index_file):
            print("Load sample indices from file")
            with open(index_file, 'rb') as f:
                subset_indices = pickle.load(f)
        else:
            dataset_size = len(self.dataset_train)
            indices = list(range(dataset_size))
            random.seed(2020)
            random.shuffle(indices)
            subset_indices = indices[:548]
            with open(index_file, 'wb') as f:
                pickle.dump(subset_indices, f)

        subset_dataset = torch.utils.data.Subset(self.dataset_train, subset_indices)

        sample_loader = CustomDataLoader(
            config=self.config,
            dataset=subset_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg,
        )
        return sample_loader

    def calc_FLOPs(self):
        sample_loader = CustomDataLoader(
                config = self.config,
                dataset=self.dataset_valid,
                batch_size=1,
                shuffle=False,
                num_workers=self.config.WORKERS,
                drop_last=False,
                collate_fn=collate_fn_mmg
            )
        loader = iter(sample_loader)
        item = next(loader)
        
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(item)
        
        if self.model_name =='SGGpoint':
            edge_indices = edge_indices.t()
            print('edge_indices shape: ', edge_indices.shape)
              
        inputs = (obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids, False)
        #kwargs = {'descriptor': descriptor, 'batch_ids':batch_ids,'istrain': False}
        return FlopCountAnalysis(self.model, inputs)
    
    def recover_submodules(self):
        exp = self.config.exp
        # submodules recovery for baseline model KD
        if '_b_' in exp:
            self.config.MODEL.N_LAYERS = self.config.MODEL.N_LAYERS * 2
            if self.model_name == 'sgpn':
                self.config.MODEL.N_LAYERS +=1
        if '_p_' in exp:
            self.config.MODEL.point_feature_size *= 2
        if '_e_' in exp:
            self.config.MODEL.edge_feature_size *= 2
        if '_a_' in exp:
            self.config.MODEL.DIM_ATTEN *= 2

        # 최종 설정 출력
        print(f"Updated Config: block: {self.config.MODEL.N_LAYERS}, point_dim: {self.config.MODEL.point_feature_size}, edge_dim: {self.config.MODEL.edge_feature_size}, attention_dim: {self.config.MODEL.DIM_ATTEN}")
    
    def remove_submodules(self):
        exp = self.config.exp
        # submodules recovery for baseline model KD
        if '_b_' in exp:
            self.config.MODEL.N_LAYERS = int(self.config.MODEL.N_LAYERS / 2)
        if '_p_' in exp:
            self.config.MODEL.point_feature_size = int(self.config.MODEL.point_feature_size / 2)
        if '_e_' in exp:
            self.config.MODEL.edge_feature_size = int(self.config.MODEL.edge_feature_size  / 2)
        if '_a_' in exp:
            self.config.MODEL.DIM_ATTEN = int(self.config.MODEL.DIM_ATTEN / 2)

        # 최종 설정 출력
        print(f"Updated Config: block: {self.config.MODEL.N_LAYERS}, point_dim: {self.config.MODEL.point_feature_size}, edge_dim: {self.config.MODEL.edge_feature_size}, attention_dim: {self.config.MODEL.DIM_ATTEN}")

    def get_nested_attr(self, obj, attr_path):
            attrs = attr_path.split('.')
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
    
    def get_ignore_layer_dimensions(self):
        if self.model_name in ['sgfn', 'sgfnattn', 'Mmgnet']:
            ignore_dims = [self.config.MODEL.point_feature_size, self.config.MODEL.edge_feature_size]
            print(f'ignore_dims: {ignore_dims}')
        elif self.model_name == 'sgpn':
            ignore_dims = [self.config.MODEL.point_feature_size, self.config.MODEL.DIM_ATTEN*2+self.config.MODEL.edge_feature_size]
        return ignore_dims

    def gcn_pruning(self, debug_mode = False):
        print(f'=== {self.model_name} (GCN) Structured Pruning   ===')
        prun_type = "gcn"
        
        if self.model_name in ['sgfn','sgfnattn']:
            before_point_params = self.count_parameters(self.model.gcn.gconvs[0].prop)
            before_edge_params = self.count_parameters(self.model.gcn.gconvs[0].edgeatten.nn_edge)
            example_inputs_nn_edge  = torch.randn(1, self.mconfig.point_feature_size * 2 + self.mconfig.edge_feature_size).to(self.config.DEVICE)
            example_inputs_prop  = torch.randn(1, self.mconfig.point_feature_size + self.mconfig.DIM_ATTEN).to(self.config.DEVICE)
            
            self.obj_ratio, self.rel_ratio = mlp_importance_estimator(self.baseline_acc[0], self.baseline_acc[1], self.phase1_acc[0], self.phase1_acc[1],
                                                                       before_point_params, before_edge_params, total_pruning_ratio=self.st_pruning_ratio)
            ignore_dims = self.get_ignore_layer_dimensions()
            print(f'ignore_dims: {ignore_dims}')

            for idx in range(0,self.config.MODEL.N_LAYERS):
                module = getattr(self.model.gcn, "gconvs")[idx]
                for mlp in ['prop', 'edgeatten.nn_edge']:
                    if mlp == 'prop':
                        example_inputs = (example_inputs_prop,)
                        max_ratio = self.obj_ratio
                    else:
                        example_inputs = (example_inputs_nn_edge,)
                        max_ratio = self.rel_ratio
                    mlp_module = self.get_nested_attr(module, mlp)
            
                    base_ops, origin_params_count = tp.utils.count_ops_and_params(mlp_module, example_inputs=example_inputs)
                    print(f'gconvs[{idx}]_{mlp} base_ops: {base_ops}, origin_params_count: {origin_params_count}')
                    
                    gcn_3ds_pruner = self.get_pruner(mlp_module, example_inputs=example_inputs, num_classes=ignore_dims)
                    
                    pruned_ratio = 0  # Initialize pruned_ratio before the loop
                    while pruned_ratio < max_ratio:
                            gcn_3ds_pruner.step()
                            pruned_ops, params_count = tp.utils.count_ops_and_params(mlp_module, example_inputs=example_inputs)
                            pruned_ratio = (origin_params_count - params_count) / origin_params_count
                            current_speed_up = float(base_ops) / pruned_ops
                            if gcn_3ds_pruner.current_step == gcn_3ds_pruner.iterative_steps:
                                break

                    print(f'en_current_speed_up: {current_speed_up}, pruned_ratio: {pruned_ratio}\n')
            after_point_params = self.count_parameters(self.model.gcn.gconvs[0].prop)
            after_edge_params = self.count_parameters(self.model.gcn.gconvs[0].edgeatten.nn_edge)
            
                    #self.gnn_pruned_ratio = self.go_prune(prun_type, "gconvs",gcn_3ds_pruner, example_inputs, base_ops, origin_params_count, idx)
        
        elif self.model_name == 'sgpn':
            before_point_params = self.count_parameters(self.model.gcn.gconvs[0].nn2)
            before_edge_params = self.count_parameters(self.model.gcn.gconvs[0].nn1)
            example_inputs_nn1  = torch.randn(1, self.mconfig.point_feature_size * 2 + self.mconfig.edge_feature_size).to(self.config.DEVICE)
            example_inputs_nn2  = torch.randn(1, self.mconfig.DIM_ATTEN).to(self.config.DEVICE)

            ignore_dims = self.get_ignore_layer_dimensions()
            print(f'ignore_dims: {ignore_dims}')

            self.obj_ratio, self.rel_ratio = mlp_importance_estimator(self.baseline_acc[0], self.baseline_acc[1], self.phase1_acc[0], self.phase1_acc[1],
                                                                       before_point_params, before_edge_params, total_pruning_ratio=self.st_pruning_ratio)
            for idx in range(0,self.config.MODEL.N_LAYERS):
                module = getattr(self.model.gcn, "gconvs")[idx]
                for mlp in ['nn1', 'nn2']:
                    if mlp == 'nn2':
                        example_inputs = (example_inputs_nn2,)
                        max_ratio = self.obj_ratio
                    else:
                        example_inputs = (example_inputs_nn1,)
                        max_ratio = self.rel_ratio
                    mlp_module = self.get_nested_attr(module, mlp)
            
                    base_ops, origin_params_count = tp.utils.count_ops_and_params(mlp_module, example_inputs=example_inputs)
                    print(f'gconvs[{idx}]_{mlp} base_ops: {base_ops}, origin_params_count: {origin_params_count}')
                    
                    gcn_3ds_pruner = self.get_pruner(mlp_module, example_inputs=example_inputs, num_classes=ignore_dims)
                    
                    pruned_ratio = 0  # Initialize pruned_ratio before the loop
                    while pruned_ratio < max_ratio:
                            gcn_3ds_pruner.step()
                            pruned_ops, params_count = tp.utils.count_ops_and_params(mlp_module, example_inputs=example_inputs)
                            pruned_ratio = (origin_params_count - params_count) / origin_params_count
                            current_speed_up = float(base_ops) / pruned_ops
                            if gcn_3ds_pruner.current_step == gcn_3ds_pruner.iterative_steps:
                                break

                    print(f'current_speed_up: {current_speed_up}, pruned_ratio: {pruned_ratio}\n')
            after_point_params = self.count_parameters(self.model.gcn.gconvs[0].nn1)
            after_edge_params = self.count_parameters(self.model.gcn.gconvs[0].nn2)
        
        
        elif self.model_name == 'SGGpoint':

            module = getattr(self.model, "gcn")
            before_point_params = self.count_parameters(module)
            before_edge_params = self.count_parameters(module)
            
            num_nodes, num_edges = 136, 1048
            node_features_example = torch.randn(num_nodes, self.mconfig.point_feature_size).to(self.config.DEVICE)
            edge_features_example = torch.randn(num_edges, self.mconfig.edge_feature_size).to(self.config.DEVICE)
            edge_indices_example = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long).to(self.config.DEVICE)
            example_inputs = (node_features_example, edge_features_example, edge_indices_example)
            
            
            ignore_layers = [module.obj_mlp, module.rel_mlp, module.node_attentionND, module.edge_attentionND, module.edge_MLP2]
            print(f'ignore_layers: {ignore_layers}')

            base_ops, origin_params_count = tp.utils.count_ops_and_params(module, example_inputs=example_inputs)
            print(f'gconvs_base_ops: {base_ops}, origin_params_count: {origin_params_count}')
            
            gcn_pruner = self.get_pruner(module, example_inputs=example_inputs, num_classes=[], ignored_layers=ignore_layers)

            pruned_ratio = 0  # Initialize pruned_ratio before the loop
            while pruned_ratio < self.st_pruning_ratio:
                gcn_pruner.step()
                pruned_ops, params_count = tp.utils.count_ops_and_params(module, example_inputs=example_inputs)
                pruned_ratio = (origin_params_count - params_count) / origin_params_count
                current_speed_up = float(base_ops) / pruned_ops
                if gcn_pruner.current_step == gcn_pruner.iterative_steps:
                        break

            print(f'current_speed_up: {current_speed_up}, pruned_ratio: {pruned_ratio}\n')
            after_point_params = self.count_parameters(module)
            after_edge_params = self.count_parameters(module)

        elif self.model_name == 'imp':
            before_point_params = self.count_parameters(self.model.gcn)
            before_edge_params = before_point_params

            num_nodes, num_edges = 136, 1048
            node_features_example = torch.randn(num_nodes, self.mconfig.point_feature_size).to(self.config.DEVICE)
            edge_features_example = torch.randn(num_edges, self.mconfig.edge_feature_size).to(self.config.DEVICE)
            edge_indices_example = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long).to(self.config.DEVICE)
            example_inputs = (node_features_example, edge_features_example, edge_indices_example)

            ignore_layers =[]
            base_ops, origin_params_count = tp.utils.count_ops_and_params(self.model.gcn, example_inputs=example_inputs)
            print(f'gconvs base_ops: {base_ops}, origin_params_count: {origin_params_count}')
            
            gcn_3ds_pruner = self.get_pruner(self.model.gcn, example_inputs=example_inputs, num_classes=[], ignored_layers=ignore_layers)
            pruned_ratio = 0  # Initialize pruned_ratio before the loop
            while pruned_ratio < self.st_pruning_ratio:
                gcn_3ds_pruner.step()
                pruned_ops, params_count = tp.utils.count_ops_and_params(self.model.gcn, example_inputs=example_inputs)
                pruned_ratio = (origin_params_count - params_count) / origin_params_count
                current_speed_up = float(base_ops) / pruned_ops
                if gcn_3ds_pruner.current_step == gcn_3ds_pruner.iterative_steps:
                    break

            print(f'current_speed_up: {current_speed_up}, pruned_ratio: {pruned_ratio}\n')
            after_point_params = before_point_params
            after_edge_params = before_point_params

        ## vl-sat mmg pruning
        else:
            before_point_params = self.count_parameters(self.model.mmg.gcn_3ds[0].prop)
            before_edge_params = self.count_parameters(self.model.mmg.gcn_3ds[0].edgeatten.nn_edge)
            example_inputs_nn_edge  = torch.randn(1, self.mconfig.point_feature_size * 2 + self.mconfig.edge_feature_size).to(self.config.DEVICE)
            example_inputs_prop  = torch.randn(1, self.mconfig.point_feature_size + self.mconfig.DIM_ATTEN).to(self.config.DEVICE)
            
            ignore_dims = self.get_ignore_layer_dimensions()
            print(f'ignore_dims: {ignore_dims}')
            # acc
            self.obj_ratio, self.rel_ratio = mlp_importance_estimator(self.baseline_acc[0], self.baseline_acc[1], self.phase1_acc[0], self.phase1_acc[1],
                                                                       before_point_params, before_edge_params, total_pruning_ratio=self.st_pruning_ratio)
            for idx in range(0,self.config.MODEL.N_LAYERS):
                for gcn in ['gcn_2ds', 'gcn_3ds']:
                    if gcn == 'gcn_2ds':
                        module = getattr(self.model.mmg, "gcn_2ds")[idx]
                    else:
                        module = getattr(self.model.mmg, "gcn_3ds")[idx]

                    for mlp in ['prop', 'edgeatten.nn_edge']:
                        if mlp == 'prop':
                            example_inputs = (example_inputs_prop,)
                            max_ratio = self.obj_ratio
                        else:
                            example_inputs = (example_inputs_nn_edge,)
                            max_ratio = self.rel_ratio
                        mlp_module = self.get_nested_attr(module, mlp)
                
                        base_ops, origin_params_count = tp.utils.count_ops_and_params(mlp_module, example_inputs=example_inputs)
                        print(f'gcn_{gcn}[{idx}]_base_ops: {base_ops}, {gcn}[{idx}]_origin_params_count: {origin_params_count}')
                        
                        pruner = self.get_pruner(mlp_module, example_inputs=example_inputs, num_classes=ignore_dims)
                        
                        pruned_ratio = 0  # Initialize pruned_ratio before the loop
                        while pruned_ratio < max_ratio:
                            pruner.step()
                            pruned_ops, params_count = tp.utils.count_ops_and_params(mlp_module, example_inputs=example_inputs)
                            pruned_ratio = (origin_params_count - params_count) / origin_params_count
                            current_speed_up = float(base_ops) / pruned_ops
                            if pruner.current_step == pruner.iterative_steps:
                                break
                        print(f'en_current_speed_up: {current_speed_up}, pruned_ratio: {pruned_ratio}\n')

            after_point_params = self.count_parameters(self.model.mmg.gcn_3ds[0].prop)
            after_edge_params = self.count_parameters(self.model.mmg.gcn_3ds[0].edgeatten.nn_edge)
        
        print(f"Before Point mlp Parameters: {before_point_params:,}")
        print(f"Before Edge mlp Parameters: {before_edge_params:,}")
        print(f"After Point mlp Parameters: {after_point_params:,}")
        print(f"After Edge mlp Parameters: {after_edge_params:,}")
        print(f"Before After Point mlp Parameters ratio: {after_point_params/before_point_params:.2f}")
        print(f"Before After Edge mlp Parameters ratio: {after_edge_params/before_edge_params:.2f}")
        self.gnn_pruned_ratio = pruned_ratio
        
    def get_gnn_name(self):
        gnn_name = 'mmg' if self.model_name == 'Mmgnet' else 'gcn'
        sub_module = 'gcn_3ds' if self.model_name == 'Mmgnet' else 'gconvs'
        return gnn_name, sub_module
    
    def compute_taylor_scores(self, layer_type, num_samples=5):
        """한 번만 스코어 계산 후 저장"""
        sample_loader = self.get_sample_loader()
        loader = iter(sample_loader)
        scores = {}

        gnn_name, sub_module = self.get_gnn_name()
        gnn_module = getattr(self.model, gnn_name)

        print(f'===  Taylor Score Calculation Layer Type : {layer_type} ===')
        # 타겟 레이어 설정
        if layer_type == 'edge':
            if self.model_name == 'sgpn':
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].nn1[2]]
            elif self.model_name == 'SGGpoint':
                target_layers = [gnn_module.rel_mlp]
            else:
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].edgeatten.nn_edge[2]]
        elif layer_type == 'point':
            if self.model_name == 'sgpn':
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].nn2[2]]
            elif self.model_name == 'SGGpoint':
                target_layers = [gnn_module.obj_mlp]
            else:
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].prop[2]]
        elif layer_type == 'gnn':
            target_layers = [gnn_module]
        elif layer_type == 'mlp':
            
            if self.model_name == 'SGGpoint':
                target_layers = [gnn_module.edge_MLP1, gnn_module.edge_MLP2, gnn_module.node_indicator_reduction, gnn_module.rel_mlp]
            else:    
                target_layers = []
                ignore_dimension = self.get_ignore_layer_dimensions()
                for module in gnn_module.modules():
                    if isinstance(module, torch.nn.Linear)and module.out_features not in ignore_dimension:
                        target_layers.append(module)
            
            if not target_layers:
                print("GNN 모듈 내에 MLP 레이어가 없습니다.")

        print(f"Target Layers: {target_layers}")
        # 샘플 데이터로 스코어 계산
        for _ in range(num_samples):
            try:
                item = next(loader)
            except StopIteration:
                loader = iter(sample_loader)
                item = next(loader)
            obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(item)
            loss = self.model.val_loss(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids,
                                                    weights_obj=self.dataset_train.w_cls_obj, 
                                                    weights_rel=self.dataset_train.w_cls_rel,
                                                    ignore_none_rel = False)
            
            for layer_idx, target_layer in enumerate(target_layers):
                for name, param in target_layer.named_parameters():
                    if param.grad is not None:
                        score = torch.abs(param.grad * param)
                        # 키에 layer_idx 포함하여 고유하게 만듭니다.
                        key = f"layer{layer_idx}.{name}"
                        if key in scores:
                            scores[key] += score
                        else:
                            scores[key] = score.clone()
                    else:
                        print(f"{name}의 grad가 None입니다.")
                            
        # 평균 스코어 계산
        for name in scores:
            scores[name] /= num_samples
        self.scores = scores  # 스코어를 캐시에 저장
    
    def taylor_expansion_based_pruning(self, layer_type, ratio):
        

        if self.scores is None:
            raise ValueError("Taylor score가 계산되지 않았습니다. 먼저 compute_taylor_scores를 호출하세요.")

        ratio_diff = ratio - self.previous_ratio  # Current 비율과 Previous 비율의 차이 계산
        print(f"Previous Ratio: {self.previous_ratio * 100:.2f}%, Current Ratio: {ratio * 100:.2f}%, Additional Pruning: {ratio_diff * 100:.2f}%")

        # 중요도 스코어를 결합
        all_scores = torch.cat([s.flatten() for s in self.scores.values()])
        
        num_params_to_keep = int(len(all_scores) * (1 - ratio))  # 전체 파라미터 중 유지할 개수 계산
        threshold, _ = torch.topk(all_scores, num_params_to_keep, largest=True)
        acceptable_score = threshold[-1]

        gnn_name, sub_module = self.get_gnn_name()
        gnn_module = getattr(self.model, gnn_name)

        # 타겟 레이어 설정
        if layer_type == 'edge':
            if self.model_name == 'sgpn':
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].nn1[2]]
            elif self.model_name == 'SGGpoint':
                target_layers = [gnn_module.rel_mlp]
            else:
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].edgeatten.nn_edge[2]]
        elif layer_type == 'point':
            if self.model_name == 'sgpn':
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].nn2[2]]
            elif self.model_name == 'SGGpoint':
                target_layers = [gnn_module.obj_mlp]
            else:
                target_layers = [getattr(gnn_module, sub_module)[self.mconfig.N_LAYERS -1].prop[2]]
        elif layer_type == 'gnn':
            target_layers = [gnn_module]
        elif layer_type == 'mlp':
            
            if self.model_name == 'SGGpoint':
                target_layers = [gnn_module.edge_MLP1, gnn_module.edge_MLP2, gnn_module.node_indicator_reduction, gnn_module.rel_mlp]
            else:    
                target_layers = []
                ignore_dimension = self.get_ignore_layer_dimensions()
                for module in gnn_module.modules():
                    if isinstance(module, torch.nn.Linear)and module.out_features not in ignore_dimension:
                        target_layers.append(module)
            
            if not target_layers:
                print("GNN 모듈 내에 MLP 레이어가 없습니다.")

        print("Before Pruning:")

        for layer_idx, target_layer in enumerate(target_layers):
            for name, param in target_layer.named_parameters():
                # 키에 layer_idx 포함하여 고유하게 만듭니다.
                key = f"layer{layer_idx}.{name}"
                if key in self.scores:
                    mask = (self.scores[key] >= acceptable_score).float()
                    with torch.no_grad():
                        param.data *= mask  # 마스킹 적용

        self.previous_ratio = ratio  # 현재 비율을 저장
        print(f"Pruning Complete: {ratio * 100:.2f}% (Layer: {layer_type})")
        #visualize_taylor_scores(scores = scores, pruned_params= pruned_params, layer_type= layer_type)

        return self.log_sparsity(gnn_module, layer_type)
    
    def log_sparsity(self, module, name):
        total_params = 0
        non_zero_params = 0
        print(f"=== {name} Sparsity ===")
        for param_name, param in module.named_parameters():
            if 'weight' in param_name:  # 중요한 레이어만 선택
                total_params += param.numel()
                non_zero_params += param.nonzero().size(0)
        sparsity = 100 * (1 - non_zero_params / total_params)
        print(f"Sparsity of {name}: {sparsity:.2f}%")
        return sparsity

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_submodule_parameters(self,model):
        submodule_params = {}
        print (f'=== {self.model_name} Submodule Parameters ===')
        for name, module in model.named_children():
            submodule_params[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: {submodule_params[name]:,}")
        return submodule_params
    
    
    def track_pruned_weights(self):
        for name, module in getattr(self.model, 'mmg').named_modules():
            if name in self.masks:
                mask = self.masks[name]
                weight = module.weight.data
                pruned_weights = weight[mask == 0]
                print(f"Module: {name} | Pruned weights mean: {pruned_weights.mean().item()} | std: {pruned_weights.std().item()}")
            # else:
            #     print("No pruned weights found in the model.")
    

    def validation(self, debug_mode = False, sample_evaluation = False):
        
        if self.model.config.EVAL == True and sample_evaluation == False:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            result_print = Result_print(save_path, self.config.exp, self.dataset_valid.classNames,
                                        self.dataset_valid.relationNames, use_rio27=False)
        else:
            result_print = None    
        if sample_evaluation:
            val_loader = self.get_sample_loader()
        else:
            val_loader = CustomDataLoader(
                config = self.config,
                dataset=self.dataset_valid,
                batch_size=1,
                shuffle=False,
                num_workers=self.config.WORKERS,
                drop_last=False,
                collate_fn=collate_fn_mmg
            )

        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array([]), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        topk_obj_2d_list, topk_rel_2d_list, topk_triplet_2d_list = np.array([]), np.array([]), np.array([])

        total_inference_time = 0
        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            # Start timing
            start_time = time.time()
            
            with torch.no_grad():
                if self.model.config.EVAL and sample_evaluation == False:
                    obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_id, split_id, origin_obj_points = self.data_processing_val(items)
                    
                    top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                        = self.model.process_val(result_print, obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, scan_id, split_id, origin_obj_points, batch_ids, use_triplet=True)
                else:
                    obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, _, _, _ = self.data_processing_val(items)

                    top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                        = self.model.process_val(result_print, obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids= batch_ids, use_triplet=True)
            # End timing
            end_time = time.time()
            total_inference_time += end_time - start_time


            ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_obj_2d_list = np.concatenate((topk_obj_2d_list, top_k_obj_2d))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_rel_2d_list = np.concatenate((topk_rel_2d_list, top_k_rel_2d))
            topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            topk_triplet_2d_list = np.concatenate((topk_triplet_2d_list, top_k_2d_triplet))
            if cls_matrix is not None:
                cls_matrix_list.extend(cls_matrix)
                sub_scores_list.extend(sub_scores)
                obj_scores_list.extend(obj_scores)
                rel_scores_list.extend(rel_scores)

            
            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/obj_cls_2d_acc", (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_2d_acc", (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_2d_acc", (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@1/rel_cls_2d_acc", (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_2d_acc", (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_2d_acc", (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@50/triplet_2d_acc", (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)),
                    ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@100/triplet_2d_acc", (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)),]

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])

        if self.model.config.EVAL and sample_evaluation == False:
            result_print.print_result(self.config.exp)

        cls_matrix_list = np.stack(cls_matrix_list)
        sub_scores_list = np.stack(sub_scores_list)
        obj_scores_list = np.stack(obj_scores_list)
        rel_scores_list = np.stack(rel_scores_list)
        mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        mean_recall_2d = get_mean_recall(topk_triplet_2d_list, cls_matrix_list)
        zero_shot_recall, non_zero_shot_recall, all_zero_shot_recall = get_zero_shot_recall(topk_triplet_list, cls_matrix_list, self.dataset_valid.classNames, self.dataset_valid.relationNames)
        
        if self.model.config.EVAL:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path,'topk_pred_list.npy'), topk_rel_list )
            np.save(os.path.join(save_path,'topk_triplet_list.npy'), topk_triplet_list)
            np.save(os.path.join(save_path,'cls_matrix_list.npy'), cls_matrix_list)
            np.save(os.path.join(save_path,'sub_scores_list.npy'), sub_scores_list)
            np.save(os.path.join(save_path,'obj_scores_list.npy'), obj_scores_list)
            np.save(os.path.join(save_path,'rel_scores_list.npy'), rel_scores_list)
            result_path = os.path.join(save_path, 'result.txt')
            f_in = open(result_path, 'w')
            self.end_time = time.time()
        
        else:
            f_in = None   
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_1 = (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_5 = (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_10 = (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_1 = (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_3 = (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_5 = (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)
        triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        triplet_acc_2d_50 = (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)
        triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
        triplet_acc_2d_100 = (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)

        rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
        rel_acc_2d_mean_1, rel_acc_2d_mean_3, rel_acc_2d_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_2d_list)
        
        ## save results
        print("\n---  Print Evaluation Results  ---")
        print(f"Experiment: {self.exp}", file=f_in)
        print(f"Model : {self.model_name}")
        print(f"Parameter reduction part: {self.config.pruning_part}", file=f_in)
        
        if self.st_pruning_ratio:
            print(f"Structured Pruning ratio setting: {self.st_pruning_ratio}", file=f_in)
            if self.encoder_pruned_ratio:
                print(f"Encoder Structured Pruning ratio: {self.encoder_pruned_ratio}", file=f_in)
            if self.gnn_pruned_ratio:
                print(f"GNN Structured Pruning ratio: {self.gnn_pruned_ratio}", file=f_in)
            if self.classifier_pruned_ratio:    
                print(f"Classifier Structured Pruning ratio: {self.classifier_pruned_ratio}", file=f_in)
            print(f'obj_ratio, rel_ratio: {self.obj_ratio, self.rel_ratio}', file=f_in)
        if self.unst_pruning_ratio:            
            print(f"Unstructured Pruning ratio setting: {self.unst_pruning_ratio}", file=f_in)

        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)   
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in) 
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)  
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in) 
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)   
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)   
        print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in) 
        print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in) 
        print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
        print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
        print(f"Eval: 3d zero-shot recall@50 : {zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d zero-shot recall@100: {zero_shot_recall[1]}", file=f_in)
        print(f"Eval: 3d non-zero-shot recall@50 : {non_zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d non-zero-shot recall@100: {non_zero_shot_recall[1]}", file=f_in)
        print(f"Eval: 3d all-zero-shot recall@50 : {all_zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d all-zero-shot recall@100: {all_zero_shot_recall[1]}", file=f_in)
        
        #print(f"Eval: 2d obj Acc@1: {obj_acc_2d_1}", file=f_in)
        #print(f"Eval: 2d obj Acc@5: {obj_acc_2d_5}", file=f_in)  
        #print(f"Eval: 2d obj Acc@10: {obj_acc_2d_10}", file=f_in)
        #print(f"Eval: 2d rel Acc@1: {rel_acc_2d_1}", file=f_in)
        #print(f"Eval: 2d rel Acc@3: {rel_acc_2d_3}", file=f_in)
        #print(f"Eval: 2d rel Acc@5: {rel_acc_2d_5}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@1: {rel_acc_2d_mean_1}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@3: {rel_acc_2d_mean_3}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@5: {rel_acc_2d_mean_5}", file=f_in)
        #print(f"Eval: 2d triplet Acc@50: {triplet_acc_2d_50}", file=f_in)
        #print(f"Eval: 2d triplet Acc@100: {triplet_acc_2d_100}", file=f_in)
        #print(f"Eval: 2d mean recall@50: {mean_recall_2d[0]}", file=f_in)
        #print(f"Eval: 2d mean recall@100: {mean_recall_2d[1]}", file=f_in)
        if self.model.config.EVAL:
            ## calculate flops
            flops = self.calc_FLOPs().total()
            flops = flops / 1e6
            print(f'\nTotal Flops: {flops:.4f} million FLOPs', file=f_in)
            
            # calculate total parameters
            total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'All Parameters: {total_param:,}', file=f_in)

            # calculate submodules parameters
            mappers = 0
            submodule_params = self.get_submodule_parameters()
            for name, params in submodule_params.items():
                if name in ['mmg', 'gcn', 'edge_gcn']:
                    print(f"gnn_total: {params:,}", file=f_in) 
                print(f"{name}: {params:,}", file=f_in)
                
                if name in ['obj_feature_dim_mapper', 'edge_feature_dim_mapper']:
                    mappers+= params
            print(f'Total Parameters: {total_param-mappers:,}', file=f_in)
            if self.config.KD.kd:
                print(f' Distillation loss rate: {self.beta[0], self.beta[1]}', file=f_in) 
            
            if self.unst_pruning_ratio:
                total_non_zero = 0
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        num_params = param.numel()
                        if name.endswith('weight'):
                            non_zero_params = torch.count_nonzero(param).item()
                        else:
                            non_zero_params = num_params
                        total_non_zero += non_zero_params
                print(f'Total Parameters after Unstructured pruning: {total_non_zero:,}', file=f_in)

            if self.config.MODE != 'eval':
                # total time
                total_time_minutes = (self.end_time - self.start_time) // 60
                total_time_hours = total_time_minutes // 60
                remaining_minutes = total_time_minutes % 60
                print(f'Total Training Time: {int(total_time_hours)} hours {int(remaining_minutes)} minutes', file=f_in)

            total_inference_time_minutes = total_inference_time // 60
            total_inference_time_hours = total_inference_time_minutes // 60
            remaining_inference_minutes = total_inference_time_minutes % 60
        
            print(f'Total Inference Time: {int(total_inference_time_hours)} hours {int(remaining_inference_minutes)} minutes', file=f_in)
            
            f_in.close()
            print("===   Evaluation done!  ===")
            
            ## save results to gspread
            flag = self.model_name
            save_gspread(result_path, flag)
            
        logs = [("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@1/obj_2d_cls_acc", obj_acc_2d_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@5/obj_2d_cls_acc", obj_acc_2d_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@10/obj_2d_cls_acc", obj_acc_2d_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@1/rel_2d_cls_acc", rel_acc_2d_1),
                ("Acc@1/rel_2d_cls_acc_mean", rel_acc_2d_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@3/rel_2d_cls_acc", rel_acc_2d_3),
                ("Acc@3/rel_2d_cls_acc_mean", rel_acc_2d_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@5/rel_2d_cls_acc", rel_acc_2d_5),
                ("Acc@5/rel_2d_cls_acc_mean", rel_acc_2d_mean_5),
                ("Acc@50/triplet_acc", triplet_acc_50),
                ("Acc@50/triplet_2d_acc", triplet_acc_2d_50),
                ("Acc@100/triplet_acc", triplet_acc_100),
                ("Acc@100/triplet_2d_acc", triplet_acc_2d_100),
                ("mean_recall@50", mean_recall[0]),
                ("mean_2d_recall@50", mean_recall_2d[0]),
                ("mean_recall@100", mean_recall[1]),
                ("mean_2d_recall@100", mean_recall_2d[1]),
                ("zero_shot_recall@50", zero_shot_recall[0]),
                ("zero_shot_recall@100", zero_shot_recall[1]),
                ("non_zero_shot_recall@50", non_zero_shot_recall[0]),
                ("non_zero_shot_recall@100", non_zero_shot_recall[1]),
                ("all_zero_shot_recall@50", all_zero_shot_recall[0]),
                ("all_zero_shot_recall@100", all_zero_shot_recall[1])
                ]
        
        self.log(logs, self.model.iteration)
        return mean_recall[0]
    
    
    def get_submodule_parameters(self):
        submodule_params = {}
        for name, module in self.model.named_children():
            submodule_params[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return submodule_params
    
    def compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
        
        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5) 
           
        predicate_mean_1 = np.mean(predicate_mean_1)
        predicate_mean_3 = np.mean(predicate_mean_3)
        predicate_mean_5 = np.mean(predicate_mean_5)

        return predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100
