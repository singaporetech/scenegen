from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import os
from generator import generate

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"

@app.post("/predictions")
async def get_image(request: PromptRequest):
    try:
        image_path = generate(request.prompt,1)
        if os.path.exists(image_path):
            image_uri = image_to_base64(image_path)
            return JSONResponse(content={"image_uri": image_uri})
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

