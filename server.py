from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes
from langchain_community.chat_models import ChatOllama


app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/prompt/playground")


llm = ChatOllama(model="EEVE-Korean-10.8B:latest")
# llm = ChatOllama(model="dnf_EEVE-10.8B:latest")
add_routes(app, llm, path="/llm")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)