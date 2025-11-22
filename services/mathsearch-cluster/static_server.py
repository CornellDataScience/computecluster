from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

app = FastAPI()

@app.get("/results/{filename}")
def get_result_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(file_path)