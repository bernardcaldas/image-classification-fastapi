from PIL import Image
from predict import read_image
from predict import transformacao
from fastapi import FastAPI, File, UploadFile
from io import BytesIO


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/files/")
async def create_file(file: bytes = File(...)):
  
    return {"file_size": len(file)}
    


@app.post("/uploadfile/")
async def create_upload_file(file: bytes = File(...)):

    # read image
    imagem = read_image(file)
    # transform and prediction 
    prediction = transformacao(imagem)

    return prediction
