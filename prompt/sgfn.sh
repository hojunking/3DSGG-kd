#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --config $3 --tconfig $4 
}
# SGFN
run_command train test_kd_kl ./config/SGFN.json ./config/mmgnet.json 
