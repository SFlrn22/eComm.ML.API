import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated
from sklearn.decomposition import PCA
import numpy as np
import sklearn
import whisper
import torch
from Helper import *
import requests
from subprocess import CalledProcessError, run


merged_dataset, books_ratings = InitializeDataset()

app = FastAPI()

security = HTTPBasic()

@app.get("/")
def root():
    return {"Message": "API Works"}

@app.get("/GetTopTen")
def GetTopTenEndpoint(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if not (Authenticate(credentials)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return GetTopTenRecommendations(merged_dataset)

@app.get("/GetRecommendations/")
def GetRecommendationsEndpoint(isbn, type, credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if not (Authenticate(credentials)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    match type:
        case "item":
            return FilteringCollaborationRecommendations(isbn, books_ratings)
        case "content":
            return ContentBasedRecommendations(isbn, merged_dataset)
        case default:
            return "No such type of recommendation supported"

@app.post("/VTT")
def VoiceToText(file: UploadFile, credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if not (Authenticate(credentials)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    content = file.file.read()
    # with open("my_file.wav", "wb") as binary_file:
    #     binary_file.write(content)
    return GetTextFromVoice(content);

@app.post("/SearchByImage")
def SearchBookByImage(file: UploadFile, credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if not (Authenticate(credentials)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    content = file.file.read()
    return SearchByImage(content)

@app.get("/TextToImage/")
def GetImageFromText(title, credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if not (Authenticate(credentials)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return TextToImage(title)

@app.post("/ImageToText")
def GetImageFromText(file: UploadFile, credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    if not (Authenticate(credentials)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    content = file.file.read()
    return ImageToText(content)
