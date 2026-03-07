from fastapi import FastAPI, UploadFile
import shutil
from app.pipeline.processor import ClassroomProcessor

app = FastAPI()
processor = ClassroomProcessor("weights/behavior.pt", "weights/handraise.pt")

@app.post("/analyze")
async def analyze(file: UploadFile):
    input_path = f"uploads/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = f"outputs/{file.filename}"
    processor.process_video(input_path, output_path)

    return {"output": output_path}