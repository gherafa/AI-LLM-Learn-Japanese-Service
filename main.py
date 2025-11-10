from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import tempfile

from modules.textGenerator import generatePracticeText

app = FastAPI(title="Japanese Speaking Coach")

app.add_middleware(
  CORSMiddleware,
  allow_origins="*",
  allow_methods="*",
  allow_headers="*",
)

# @app.post("/analyze/")
# async def analyzeVoice(file: UploadFile = File(...)):
#   with tempfile.NamedTemporaryFile(delete = False, suffix = ".wav") as tmp:
#     tmp.write(await file.read())
#     tmp_path = tmp.name
    
#   transcription = CALL_METHOD
#   correction = CORRECT_METHOD
#   # IMPLEMENTED LATER
#   # scores = score_pronounciation(tmp_path)
  
#   return {
#     "transcription": transcription,
#     "correction": correction
#     # "scores": scores
#   }
  
@app.get("/generate_text")
def generate_text(level: str = Query("basic")):
  data = generatePracticeText(level)
  
  return JSONResponse(content=data)
  
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8080)