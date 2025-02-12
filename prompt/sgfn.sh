#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}
# Test
#run_command eval param_reduction50_gcn_sgfn gnn ./config/attn_SGFN.json 0 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/param_reduction50_gcn_sgfn
#run_command eval testtest gcn ./config/SGFN.json 0 0.5 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
#run_command prune test gnn ./config/attn_SGFN.json 0.75 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/param_reduction50_gcn_sgfn
run_command train test_sgfn_b_p_e_gcn gnn ./config/SGFN.json 0 0 x
# run_command prune param50_st75_kd_t05_attn_sgfn gnn ./config/attn_SGFN2.json 0.75 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/param_reduction50_gcn_sgfn
#run_command prune param50_st75_kd_t07_attn_sgfn gnn ./config/attn_SGFN3.json 0.75 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/param_reduction50_gcn_sgfn

# SGFN
#run_command eval test gcn ./config/attn_SGFN.json 0.05 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
#run_command eval test gnn ./config/SGFN.json 0 0.05 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
#run_command prune sgfn_baseline_st35_gcn_sgfn gnn ./config/SGFN.json 0.35 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn

#run_command train redu60_attn_dim128_gcn_sgfn gcn ./config/SGFN.json 0 0 x
#run_command prune redu60_attn_dim128_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/redu60_attn_dim128_gcn_sgfn

# run_command prune redu50_st25_sgfn gcn ./config/SGFN2.json 0.25 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/real_baseline_param50_gcn_sgfn
# #run_command prune redu60_attn_dim128_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/redu60_attn_dim128_gcn_sgfn
# run_command train redu70_attn_edge_dim128_gcn_sgfn gcn ./config/SGFN3.json 0 0 x
# run_command prune redu70_attn_edge_dim128_st25_gcn_sgfn gcn ./config/SGFN3.json 0.25 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/redu70_attn_edge_dim128_gcn_sgfn

# # attn+SGFN
#run_command prune attnSGFN_baseline_st20_gcn_attn_sgfn gnn ./config/attn_SGFN.json 0.20 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command train redu55_edge_dim128_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0 0 x
# run_command prune redu55_edge_dim128_st25_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0.25 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/redu55_edge_dim128_gcn_attn_sgfn
# run_command prune redu60_attn_edge_dim128_st25_gcn_attn_sgfn gcn ./config/attn_SGFN2.json 0.25 0 /home/oi/Desktop/song/light_gnn/config/ckp/SGFN/redu60_attn_edge_dim128_gcn_attn_sgfn

