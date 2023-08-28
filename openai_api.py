# coding=utf-8
# Implements API for ChatGLM2-6B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModel
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# langchain openai need this parameter
_usage = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}

    
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]]


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Union[Literal["stop", "length"], str]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Union[Literal["stop", "length"], str]


class usage(BaseModel):
    """
    langchain openai request need this parameter
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: usage


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    print(query)
    # print(history)
    response, _ = model.chat(tokenizer, query, history=history)

    finish_reason = "stop"

    if request.stop and type(request.stop) == list:
        for _stop in request.stop:
            if _stop in response:
                response = response.split(_stop)[0]
                finish_reason = _stop

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason=finish_reason
    )

    print(choice_data)

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=_usage)

async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))


    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield '[DONE]'



class CompletionResponseChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Optional[int] = 0
    finish_reason: Union[Literal["stop", "length"], str]


class CompletionRequest(BaseModel):
    prompt: List[str]
    model: Optional[str] = "chatGLM2-6B"
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.95
    top_p: Optional[float] = 0.7
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = 0
    stop: Optional[Union[str, List[str]]]

class CompletionResponse(BaseModel):
    id: str = "chatGLM2-6B"
    model: str = "chatGLM2-6B"
    object: Union[Literal["stop", "length"], str]
    choices: List[CompletionResponseChoice]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: usage


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    global model, tokenizer
    # if request.messages[-1].role != "user":
    #     raise HTTPException(status_code=400, detail="Invalid request")
    query = request.prompt[-1]
    # max_tokens = request.max_tokens
    # temperature = request.temperature
    # top_p = request.top_p
    # n = request.n

    print(query)
    response, _ = model.chat(tokenizer, query, history=[])

    finish_reason = "stop"

    if request.stop and type(request.stop) == list:
        for _stop in request.stop:
            if _stop in response:
                response = response.split(_stop)[0]
                finish_reason = _stop

        
    choice_data = CompletionResponseChoice(
        text=response,
        finish_reason=finish_reason
    )

    print(choice_data)

    return CompletionResponse(choices=[choice_data], object=finish_reason, usage=_usage)





if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("QWen/QWen-7B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "QWen/QWen-7B-Chat",
        device_map="cuda:0",
        trust_remote_code=True,
        bf16=True,
        use_flash_attn=False
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained("QWen/QWen-7B-Chat", trust_remote_code=True)


    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
