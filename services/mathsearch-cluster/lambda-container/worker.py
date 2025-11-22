import os
import time
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the logic from your preserved file
from lambda_function import process_local_job

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Up one level from lambda-container

# Match the structure you showed in the screenshot
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

def get_pending_uuids():
    """
    Finds UUIDs that exist in INPUT but don't have results in OUTPUT.
    """
    if not os.path.exists(INPUT_DIR): return []
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    input_files = os.listdir(INPUT_DIR)
    output_files = os.listdir(OUTPUT_DIR)
    
    # Get all UUIDs from PDF files in Input
    input_uuids = set()
    for f in input_files:
        if f.endswith(".pdf"):
            uuid = f.replace(".pdf", "")
            # Only count it if the corresponding image also exists
            if f"{uuid}_image.png" in input_files:
                input_uuids.add(uuid)

    # Get all UUIDs that are already done
    processed_uuids = {f.replace("_result.json", "") for f in output_files if f.endswith(".json")}
    
    # Return differences
    return list(input_uuids - processed_uuids)

if __name__ == "__main__":
    print("--- MathSearch Cluster Worker Started ---")
    print(f"Monitoring: {INPUT_DIR}")
    
    while True:
        pending = get_pending_uuids()
        
        if not pending:
            # No work found, sleep briefly
            time.sleep(2)
            continue
        
        print(f"Found {len(pending)} pending jobs: {pending}")
        
        for uuid in pending:
            # Call the function from lambda_function.py
            process_local_job(uuid, INPUT_DIR, OUTPUT_DIR)
            
            # Optional cooldown
            time.sleep(1)