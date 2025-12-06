from flask import Flask, jsonify, send_from_directory
import json
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ""  # folder containing orca.txt, compute1.txt, etc.

MACHINES = ["orca", "compute1", "compute2", "compute3"]

app = Flask(__name__, static_folder="static")


def read_machine_data(name):
    fp = DATA_DIR / f"{name}.json"
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


# ---------- Web UI ----------

@app.route("/")
def index():
    # Serve the dashboard HTML
    return send_from_directory(app.static_folder, "index.html")


# ---------- API endpoints ----------

@app.get("/all")
def api_all():
    """Return a dict {machine_name: metrics_json} for all machines with data."""
    result = {}
    for m in MACHINES:
        data = read_machine_data(m)
        if data is not None:
            result[m] = data
    return jsonify(result)


@app.get("/<name>")
def api_single(name):
    """Return metrics for a single machine."""
    if name not in MACHINES:
        return jsonify({"error": "unknown machine"}), 404

    data = read_machine_data(name)
    if data is None:
        return jsonify({"error": "no data"}), 404

    return jsonify(data)


if __name__ == "__main__":
    print("Run this using a WSGI server such as gunicorn:")
    print("    gunicorn -w 4 -b 0.0.0.0:8000 dashboard_server:app")
