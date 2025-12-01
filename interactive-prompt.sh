#!/bin/bash
# Interactive prompt submission for vLLM

echo "vLLM Interactive Prompt"
echo "======================"

# Read multi-line-friendly prompt
echo "Enter your prompt, then press ENTER:"
read -r PROMPT

# Ask max tokens
read -r -p "Max tokens [100]: " MAX_TOKENS
MAX_TOKENS=${MAX_TOKENS:-100}

# Validate max tokens is an integer
if ! [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "Error: max tokens must be a number."
    exit 1
fi

# Escape prompt for JSON
ESCAPED_PROMPT=$(printf "%s" "$PROMPT" | sed 's/"/\\"/g')

echo ""
echo "Submitting SLURM job..."
~/computecluster/submit-prompt.sh "$ESCAPED_PROMPT" "$MAX_TOKENS"
