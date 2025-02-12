#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
if __name__ == '__main__':
    #os.sys.path.append('./pytorch_geometric/torch_geometric')
    os.sys.path.append('./src')

## select process (origin, KD, pruning : model.py)
from model.model import MMGNet
from src.utils.config import Config
from utils import util, sub_module_ratio
import torch
import argparse

def main():
    config = load_config()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU[0])
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    util.set_random_seed(config.SEED)

    ## Sub-module reduction
    config = set_config(config)
    
    if config.VERBOSE:
        print(config)
    
    model = MMGNet(config)

    save_path = os.path.join(config.PATH,'config', model.model_name, model.exp)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'config.json')
    config.DEVICE = 'cuda'
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            json.dump(config, f)
                
    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")
    
    # just for test
    if config.MODE == 'eval':
        model.config.EVAL = True
        ## Pruning & Inference
        if config.pruning_method == 'st':
            model.gcn_pruning()
        elif config.pruning_method == 'unst':
            model.apply_pruning(config.pruning_part)
        
        elif config.pruning_method == 'taylor_co1':
            results = sub_module_ratio.calculate_model_parameters(config)
            ratios = [result['reduction_ratio'] for result in results]

            model.compute_taylor_scores(layer_type='gnn', num_samples=5)
            
            baseline_config = results[0]['config']
            for idx, ratio in enumerate(ratios):
                # if idx == 0:
                #     print("Skipping initial ratio of 0%")
                #     continue
                current_config = results[idx]['config']
                exp_suffix = sub_module_ratio.generate_exp_suffix(baseline_config, current_config)
                print(f'suffix: {exp_suffix}')

                sparsity = model.taylor_expansion_based_pruning(layer_type= 'gnn', ratio = ratio / 100.0)
                model.exp = config.exp + f'{int(sparsity)}%{exp_suffix}'
                model.validation(sample_evaluation = True)
            # 95%
            sparsity = model.taylor_expansion_based_pruning(layer_type= 'gnn', ratio = 95 / 100.0)
            model.exp = config.exp + f'{int(sparsity)}%'
            model.validation(sample_evaluation = True)

            exit()
        elif config.pruning_method in ['taylor_point', 'taylor_edge']:
            layer_type = config.pruning_method.split('_')[1]

            model.compute_taylor_scores(layer_type=layer_type, num_samples=5)
            sparsity = model.taylor_expansion_based_pruning(layer_type= layer_type, ratio = 0.5)
            model.exp = config.exp + f'{int(sparsity)}%'
            model.validation(sample_evaluation = True)

            exit()
        elif config.pruning_method == 'taylor_co2':
            model.compute_taylor_scores(layer_type='mlp', num_samples=5)

            for ratio in range(10, 100, 10):
                sparsity = model.taylor_expansion_based_pruning(layer_type= 'mlp', ratio = ratio / 100.0)
                model.exp = config.exp + f'{int(sparsity)}%_r{ratio}%'
                model.validation(sample_evaluation = True)
            exit()

        ## Normal Inference with exp
        
        # pruning_result = config.exp +'pruning_test.txt'
        # model.calculate_sparsity(pruning_result)

        model.config.EVAL = True
        model.validation(sample_evaluation = True)
        exit()

    if config.MODE == 'prune':
        print('===   Start Pruning   ===')
        print("Pruning method: ", config.pruning_method)

        """ Pruning & KD """
        if config.KD.kd and config.pruning_method == 'st':
            print("KD & Structured pruninng")
            model.remove_submodules()
            if config.pruning_part == 'gnn':
                model.gcn_pruning()
            else:
                print("Error: Unknown model part specified.")
                exit()
            
            """ Structured pruning"""
        elif config.pruning_method == 'st':
            print("Pruning method: Structured pruning")
            model.gcn_pruning()
        
        submodule_params = model.get_submodule_parameters()

        for name, params in submodule_params.items():
            if name in ['mmg', 'gcn', 'edge_gcn']:
                print(f"gnn_total: {params:,}") 
            print(f"{name}: {params:,}")
        # 전체 파라미터 수 계산 및 출력
        print(f"Total Parameters: {count_parameters(model.model):,}")
        print(f'\nTotal Flops: {get_flops(model):.4f} million FLOPs')
        model.train()
        
        ## After retraining, we need to validate the model
        model.load(best=True)
        model.config.EVAL = True
        model.validation()
        
        exit()
    try:
        model.load()
    except:
        print('unable to load previous model.')

    submodule_params = model.get_submodule_parameters()

    for name, params in submodule_params.items():
        if name in ['mmg', 'gcn', 'edge_gcn']:
            print(f"gnn_total: {params:,}") 
        print(f"{name}: {params:,}")
    # 전체 파라미터 수 계산 및 출력
    #print(f"Total Parameters: {count_parameters(model.model):,}")
    
    
    print(f'\nTotal Flops: {get_flops(model):.4f} million FLOPs')
    ## WITHOUT PRUNING
    model.train()
    # we test the best model in the end
    model.config.EVAL = True
    print('start validation...')
    model.load(best=True)
    model.validation()
    
def get_flops(model):
    flops = model.calc_FLOPs().total()
    flops = flops / 1e6
    return flops

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_example.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval','prune'], help='mode. can be [train,trace,eval]',required=True)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--part', type=str)
    parser.add_argument('--st_ratio', type=str)
    parser.add_argument('--unst_ratio', type=str)
    parser.add_argument('--pretrained', type=str)


    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
    
    # load config file
    config = Config(config_path)
    
    if 'NAME' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    config.exp = args.exp
    config.pruning_part = args.part
    

    if args.pretrained:
        if os.path.exists(args.pretrained):
            print(f'===   load pretrain model: {args.pretrained}   ===')
            config.MODEL.use_pretrain = args.pretrained
        elif args.pretrained =='x':
            print('===   No pretrained weight start   ===')
        else:
            raise FileNotFoundError(f"The folder '{args.pretrained}' does not exist.")
    
    if args.st_ratio != '0':
        config.pruning.st_pruning_ratio = float(args.st_ratio)
        config.pruning_method = "st"
    else:
        ## taylor pruning
        config.pruning_method = 'none'
    
    if args.unst_ratio != '0':
        config.KD.kd = True
    
    return config

def set_config(config):
    exp = config.exp
    # submodule 조건에 따른 설정 조정
    if '_b_' in exp:
        config.MODEL.N_LAYERS = int(config.MODEL.N_LAYERS / 2)
    if '_p_' in exp:
        config.MODEL.point_feature_size = int(config.MODEL.point_feature_size / 2)
    if '_e_' in exp:
        config.MODEL.edge_feature_size = int(config.MODEL.edge_feature_size / 2)
    if '_a_' in exp:
        config.MODEL.DIM_ATTEN = int(config.MODEL.DIM_ATTEN / 2)

    # 최종 설정 출력
    print(f"Sub-module Config: block: {config.MODEL.N_LAYERS}, point_dim: {config.MODEL.point_feature_size}, edge_dim: {config.MODEL.edge_feature_size}, attention_dim: {config.MODEL.DIM_ATTEN}")
    
    if config.KD.kd:
        if 'kd_kl' in exp:
            config.KD.method = 'kl'
        elif 'kd_feature' in exp:
            config.KD.method = 'feature'
        elif 'kd_fkl' in exp:
            config.KD.method = 'fkl'
        elif 'kd_mse' in exp:
            config.KD.method = 'mse'
        elif 'kd_fmse' in exp:
            config.KD.method = 'fmse'
        print (f"KD method: {config.KD.method}")
    
    if config.MODE == 'eval':
        if '_co1_' in exp:
            config.pruning_method = 'taylor_co1'
        elif '_co2_' in exp:
            config.pruning_method = 'taylor_co2'
        elif '_sub_point_' in exp:
            config.pruning_method = 'taylor_point'
        elif '_sub_edge_' in exp:
            config.pruning_method = 'taylor_edge'

        print(f"Evaluation Method: {config.pruning_method}")

    return config



if __name__ == '__main__':
    main()
