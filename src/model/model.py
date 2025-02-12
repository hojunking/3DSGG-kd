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
from process_data.relation_distribution import Result_print
# pruning
from fvcore.nn import FlopCountAnalysis

class MMGNet():
    def __init__(self, config, tconfig):
        self.config = config
        
        
        self.model_name = self.config.NAME
        self.mconfig = mconfig = config.MODEL
        self.exp = config.exp
        self.save_res = config.EVAL
        self.update_2d = config.update_2d
        self.masks = {}
        self.start_time, self.end_time = 0, 0
        
        if tconfig != 'x':
            self.tconfig = tconfig
            self.kd = True
        
        ''' Build dataset '''
        if config.MODE  == 'train':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config,split_type='train_scans', shuffle_objs=True,
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
        
        if config.MODE  == 'train':
            self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_train) // self.config.Batch_Size)
        
        elif config.MODE  == 'eval':
            self.total = self.config.total = len(self.dataset_valid) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_valid) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_valid) // self.config.Batch_Size)

        # 모델 클래스를 딕셔너리에 매핑
        model_classes = {
            'mmgnet': Mmgnet,
            'sgfn': SGFN,
            'sgfnattn': SGFN,
            'sgpn': SGPN,
            'sggpoint': SGGpoint,
            'imp':IMP
        }

        print(f'model name : {self.model_name}')
        
        # Student model load
        if self.model_name in model_classes:
            # 모델 이름이 올바른지 확인
            model_class = model_classes[self.model_name]
            self.model = model_class(config=self.config,
                                    tconfig=self.tconfig,
                                    num_obj_class=num_obj_class,
                                    num_rel_class=num_rel_class).to(config.DEVICE)
            
            if mconfig.use_pretrain != 'x':
                print(f'load weight: {mconfig.use_pretrain}')
                self.model.load_pretrain_model(mconfig.use_pretrain, skip_names=['obj_feature_dim_mapper', 'edge_feature_dim_mapper'], is_freeze=False)
        else:
            print(f'Unknown model name: {self.model_name}')
            raise NotImplementedError
        
        # Teacher model load
        if self.kd:
            print(f'teacher model name : {self.tconfig.NAME}')
            self.tconfig.max_iteration = self.max_iteration

            
            t_model_class = model_classes[self.tconfig.NAME]
            self.t_model = t_model_class(config=self.tconfig,
                                        num_obj_class=num_obj_class,
                                        num_rel_class=num_rel_class,
                                        tconfig=None).to(config.DEVICE)
            
            if self.tconfig.MODEL.use_pretrain != 'x':
                print(f'load teacher weight: {self.tconfig.MODEL.use_pretrain}')
                self.t_model.load_pretrain_model(self.tconfig.MODEL.use_pretrain, skip_names=['obj_feature_dim_mapper', 'edge_feature_dim_mapper'], is_freeze=True)
            self.beta = 0,0
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
                if self.kd:
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
            
            ## talks
            if (self.model.epoch > 20 and 'VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and self.model.epoch % self.config.VALID_INTERVAL == 0):
                print('start validation...')
                rel_acc_val = self.validation()
                self.model.eva_res = rel_acc_val
                self.save()
            
            self.model.epoch += 1
            
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

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def validation(self, debug_mode = False):
        
        if self.model.config.EVAL == True:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            result_print = Result_print(save_path, self.config.exp, self.dataset_valid.classNames,
                                        self.dataset_valid.relationNames, use_rio27=False)
        else:
            result_print = None    
        
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
                if self.model.config.EVAL:
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

        if self.model.config.EVAL:
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
        

        ## talk
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
                if name in ['obj_feature_dim_mapper', 'edge_feature_dim_mapper']:
                    mappers+= params
            print(f'Total Parameters: {total_param-mappers:,}', file=f_in)
            
            if self.config.KD.kd:
                print(f' Distillation loss rate: {self.beta[0], self.beta[1]}', file=f_in) 
            
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
