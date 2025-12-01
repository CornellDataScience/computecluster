# vLLM on the Compute Cluster

## Overview

This cluster runs vLLM (high-performance LLM inference) with SLURM job scheduling. The setup uses a client-server architecture where a persistent vLLM server runs on GPU nodes and clients submit inference requests via SLURM jobs.

## Architecture

- **Server**: vLLM OpenAI-compatible API server running on GPU nodes (port 8000)
- **Model**: Currently serving `meta-llama/Llama-2-7b-chat-hf` with 2-GPU tensor parallelism
- **Job Scheduler**: SLURM manages GPU allocation and job queuing
- **Interface**: REST API with automatic job script generation

## Quick Start

### 1. Start the vLLM Server

```bash
sbatch ~/vllm-server.sh
```

Wait 1-3 minutes for the model to load. Check if ready:

```bash
curl http://compute1:8000/v1/models
```

### 2. Submit Prompts

```bash
cd ~/computecluster

# Simple prompt
./submit_prompt.sh "What is machine learning?"

# With custom token limit
./submit_prompt.sh "Write a story about robots" 500

# Interactive mode
./interactive_prompt.sh
```

### 3. View Results

```bash
# Check job status
squeue -u $USER

# View output (after job completes)
cat ~/computecluster/jobs/prompts/vllm_prompt_*_output.json | jq .

# Monitor live
tail -f ~/computecluster/jobs/prompts/vllm_prompt_*.out
```

## Directory Structure

```
~/computecluster/
├── submit_prompt.sh          # Main prompt submission script
├── interactive_prompt.sh     # Interactive prompt interface
├── server.sh                 # vLLM server startup script
├── client.sh                 # Benchmarking script
├── jobs/prompts/             # Generated job scripts & outputs
└── vllm/                     # vLLM configurations

~/
├── vllm-server.sh            # SLURM job script for server
└── vllm-client.sh            # Simple curl-based client
```

## Key Scripts

### `vllm-server.sh`
SLURM job that launches the vLLM server with 2 GPUs. Only one server instance should run at a time.

### `submit_prompt.sh`
Automatically generates and submits SLURM jobs for inference requests. Handles:
- Job script generation
- Prompt escaping and formatting
- Server health checking
- Output file management

### `server.sh`
Configures the vLLM server with:
- Tensor parallelism across 2 GPUs
- Chunked prefill for efficiency
- Optimized batching (512 tokens/sequences)
- 8GB swap space

## Using Different Models

### 1. Download a New Model

Create `download.py` or use the existing one:

```python
from huggingface_hub import snapshot_download

model_name = "meta-llama/Llama-3-8b-Instruct"
snapshot_download(repo_id=model_name, local_dir=f"./models/{model_name}")
```

Run it (requires HuggingFace token for gated models):

```bash
conda activate vllm
python download.py
```

### 2. Update Server Configuration

Edit `~/computecluster/server.sh`:

```bash
# Change the --model parameter
--model meta-llama/Llama-3-8b-Instruct \
```

Also update `submit_prompt.sh` to use the new model name in the API call.

### 3. Restart the Server

```bash
scancel -u $USER  # Kill existing jobs
sbatch ~/vllm-server.sh
```

## Common Tasks

### Check Server Status
```bash
squeue | grep my_job_n
curl http://compute1:8000/v1/models
```

### Kill All Jobs
```bash
scancel -u $USER
```

### Benchmark Performance
```bash
cd ~/computecluster
conda activate vllm
./client.sh  # Runs 100 prompts at 10 req/s
```

### View Job Logs
```bash
# Server logs
tail -f ~/slurm-*.out

# Prompt job logs
tail -f ~/computecluster/jobs/prompts/*.out
```

## Configuration Parameters

Key vLLM server settings in `server.sh`:

- `--tensor-parallel-size 2`: Split model across 2 GPUs
- `--max-num-batched-tokens 512`: Batch size for throughput
- `--swap-space 8`: CPU offloading capacity (GB)
- `--enable-chunked-prefill`: Process long prompts efficiently

Adjust based on your hardware and workload.

## Troubleshooting

**Prompt jobs hanging**: Server isn't running. Start with `sbatch ~/vllm-server.sh`

**Out of memory**: Reduce `--max-num-batched-tokens` or use smaller model

**Port already in use**: Another server is running. Check with `squeue` and kill if needed

**Model not found**: Ensure model is downloaded and path is correct in `server.sh`

## Notes

- The server reserves 2 GPUs - plan accordingly for multi-user scenarios
- Jobs timeout after 10 minutes by default (configurable in `submit_prompt.sh`)
- All outputs are saved to `~/computecluster/jobs/prompts/` with timestamps
- SLURM warnings about `ControlMachine` can be ignored - they don't affect functionality
