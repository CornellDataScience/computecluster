#!/bin/bash
#SBATCH --job-name=vllm_prompt
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=compute
#SBATCH --account=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0   # Client does NOT need GPUs

# Load JSON tool
module load jq >/dev/null 2>&1 || true

echo "Submitting prompt to vLLM server:"
echo "explain quantum entanglement in one sentence"
echo "-------------------------------"

curl -s http://compute1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
            \"model\": \"meta-llama/Llama-2-7b-chat-hf\",
            \"prompt\": \"explain quantum entanglement in one sentence\",
            \"max_tokens\": 200
        }" | jq .

