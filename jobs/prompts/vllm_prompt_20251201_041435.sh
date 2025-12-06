#!/bin/bash
#SBATCH --output=/home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_041435.out
#SBATCH --error=/home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_041435.err
#SBATCH --job-name=vllm_prompt_20251201_041435
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --account=NONE

# Wait for server to be available
until curl -s http://compute1:8000/v1/models > /dev/null 2>&1; do
    echo "Waiting for vLLM server..."
    sleep 2
done

# Send the prompt
curl -s http://compute1:8000/v1/completions   -H "Content-Type: application/json"   -d '{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "$PROMPT\n",
        "max_tokens": 100
      }' | jq . > /home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_041435_output.json

echo "Results saved to: /home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_041435_output.json"
