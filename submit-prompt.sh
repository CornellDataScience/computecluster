#!/bin/bash
# ~/computecluster/submit_prompt.sh

if [ -z "$1" ]; then
    echo "Usage: ./submit_prompt.sh \"Your prompt here\" [max_tokens]"
    exit 1
fi

PROMPT="$1"
MAX_TOKENS="${2:-100}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="vllm_prompt_${TIMESTAMP}"
JOB_DIR="$HOME/computecluster/jobs/prompts"
JOB_SCRIPT="${JOB_DIR}/${JOB_NAME}.sh"
OUTPUT_FILE="${JOB_DIR}/${JOB_NAME}_output.json"

mkdir -p "$JOB_DIR"

# Escape quotes only
ESCAPED_PROMPT=$(printf '%s' "$PROMPT" | sed 's/"/\\"/g')

# Generate SLURM script
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --output=${JOB_DIR}/${JOB_NAME}.out
#SBATCH --error=${JOB_DIR}/${JOB_NAME}.err
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --account=NONE

# Wait for server
until curl -s http://compute1:8000/v1/models >/dev/null 2>&1; do
    echo "Waiting for vLLM server..."
    sleep 2
done

# Send request (FIXED JSON!)
curl -s http://compute1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"meta-llama/Llama-2-7b-chat-hf\", \"prompt\": \"${ESCAPED_PROMPT}\", \"max_tokens\": ${MAX_TOKENS}}" \
  | jq . > ${OUTPUT_FILE}

echo "Saved to ${OUTPUT_FILE}"
EOF

chmod +x "$JOB_SCRIPT"

JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
echo "Job submitted: $JOB_ID"
echo "Job script: $JOB_SCRIPT"
echo "Output will be saved to: $OUTPUT_FILE"
echo "Monitor with: tail -f ${JOB_DIR}/${JOB_NAME}.out"
