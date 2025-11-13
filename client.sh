#!/bin/bash

python3 -m vllm.benchmarks.serve \
  --server-url http://localhost:8000 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset dummy \
  --num-prompts 100 \
  --request-rate 10 \
  --long-prompts 0 \
  --long-prompt-len 32000

