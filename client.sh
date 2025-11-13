#!/bin/bash

python3 -m vllm.benchmarks.serve \
  --server-url http://localhost:8000 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset dummy \
  --num-prompts 50 \
  --request-rate 5 \
  --max-num-prompts 50 \
  --timeout 200 \
  --output-file results.json
