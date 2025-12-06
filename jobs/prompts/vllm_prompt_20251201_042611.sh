#!/bin/bash
#SBATCH --output=/home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_042611.out
#SBATCH --error=/home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_042611.err
#SBATCH --job-name=vllm_prompt_20251201_042611
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --account=NONE

echo "Waiting for vLLM server..."
until curl -s http://compute1:8000/v1/models > /dev/null 2>&1; do
    sleep 2
done

curl -s http://compute1:8000/v1/completions   -H "Content-Type: application/json"   -d "{
        \\"model\\": \\"meta-llama/Llama-2-7b-chat-hf\\",
        \\"prompt\\": \\"What is the capital of France?\\",
        \\"max_tokens\\": 100
      }" | jq . > /home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_042611_output.json

echo "Output saved to: /home/cadmin/computecluster/jobs/prompts/vllm_prompt_20251201_042611_output.json"
