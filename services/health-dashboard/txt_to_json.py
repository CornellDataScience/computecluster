import json
import ast
from pathlib import Path

def convert_txt_files_to_json(directory: str = "."):
    base = Path(directory)

    for txt_path in base.glob("*.txt"):
        print(f"Processing {txt_path} ...")

        text = txt_path.read_text()

        # Try JSON first
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # If it's actually a Python dict repr (old format), handle that too
            try:
                data = ast.literal_eval(text)
            except Exception as e:
                print(f"  !! Failed to parse {txt_path.name}: {e}")
                continue

        json_path = txt_path.with_suffix(".json")
        with json_path.open("w") as f:
            json.dump(data, f, indent=2)

        print(f"  -> wrote {json_path.name}")

if __name__ == "__main__":
    convert_txt_files_to_json(".")
