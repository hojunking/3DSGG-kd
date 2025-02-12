 #!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7

}

run_command train test_sgpn_b_p_e_a_gcn gnn ./config/IMP.json 0 0 x
# run_command prune param40_num_block3_25_gcn_sgpn gcn ./config/SGPN.json 0.25 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command prune param40_num_block3_50_gcn_sgpn gcn ./config/SGPN.json 0.5 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command prune param40_num_block3_75_gcn_sgpn gcn ./config/SGPN.json 0.75 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_sgpn_baseline_sgpn gcn ./config/SGPN.json 0 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# ## st pruning inference 5 - 75
# run_command eval eval_param40_st05_gcn_sgpn gcn ./config/SGPN.json 0.05 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st10_gcn_sgpn gcn ./config/SGPN.json 0.1 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st15_gcn_sgpn gcn ./config/SGPN.json 0.15 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st20_gcn_sgpn gcn ./config/SGPN.json 0.20 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st25_gcn_sgpn gcn ./config/SGPN.json 0.25 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st65_gcn_sgpn gcn ./config/SGPN.json 0.65 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st70_gcn_sgpn gcn ./config/SGPN.json 0.70 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn