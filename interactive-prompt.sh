#!/bin/bash
# Interactive prompt submission for vLLM

echo "vLLM Interactive Prompt"
echo "======================"

echo "Enter your prompt, then press ENTER:"
read -r PROMPT

read -r -p "Max tokens [100]: " MAX_TOKENS
MAX_TOKENS=${MAX_TOKENS:-100}

if ! [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "Error: max tokens must be a number."
    exit 1
fi

ESCAPED_PROMPT=$(printf "%s" "$PROMPT" | sed 's/"/\\"/g')

echo ""
echo "Submitting SLURM job..."
~/computecluster/submit-prompt.sh "$ESCAPED_PROMPT" "$MAX_TOKENS"
