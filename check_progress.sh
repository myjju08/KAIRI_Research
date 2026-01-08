#!/bin/bash
echo "=== Layer Routing Evaluation Progress ==="
echo ""
echo "Active processes: $(ps aux | grep 'main.py.*target=' | grep -v grep | wc -l)"
echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{printf "  GPU %s: %s%% util, %sMB / %sMB\n", $1, $2, $3, $4}'
echo ""
echo "Completed lr_on targets:"
ls -1d logs/lr_ablation/lr_on/target=*/guidance_name*/finished_sampling 2>/dev/null | wc -l
echo "/10"
echo ""
echo "Completed lr_off targets:"
ls -1d logs/lr_ablation/lr_off/target=*/guidance_name*/finished_sampling 2>/dev/null | wc -l
echo "/10"
echo ""
echo "Last log entries:"
tail -3 logs/lr_ablation_run.log
