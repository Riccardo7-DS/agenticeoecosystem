python scripts/run_generate.py \
    --label "single-agent-adk" \
    --model EVE-Instruct --provider eve \
    --pipeline single_adk \
    --tasks-file benchmark_tasks.json \
    --task-labels "Tool Grounding,Sequential Reasoning,Cross-agent Synthesis,Scope Awareness" \
    --output-dir output/geobenchX/run_$(date +%Y%m%d_%H%M%S)


python scripts/run_sample.py --tasks-file benchmark_tasks.json \
    --pipeline single_adk --n 5 --random --seed 42