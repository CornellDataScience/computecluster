from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mathsearch.cornelldata.science"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/results/{filename}")
def get_result_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    print("searching for file:", file_path)

    if not os.path.exists(file_path):
        # Let FastAPI generate a normal error response (will get CORS via middleware)
        raise HTTPException(status_code=404, detail="File not found")

    # File exists – return it with explicit CORS headers
    response = FileResponse(file_path, media_type="application/pdf")
    response.headers["Access-Control-Allow-Origin"] = "https://mathsearch.cornelldata.science"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), uuid: str = None, type: str = None):
    """
    Save uploaded files to the input directory.
    Accepts: file, uuid, type (either "pdf" or "image")
    """
    try:
        # Determine filename based on type
        if type == "pdf":
            filename = f"{uuid}_pdf.pdf"
        elif type == "image":
            filename = f"{uuid}_image.png"
        else:
            filename = file.filename
        
        # Save to input directory
        file_location = os.path.join(INPUT_DIR, filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "filename": filename}
    except Exception as e:
        return {"status": "error", "error": str(e)}