import base64
from typing import List

from fastapi import FastAPI, UploadFile, File
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()