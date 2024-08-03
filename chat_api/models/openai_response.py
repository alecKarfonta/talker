from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime
import json

class Message(BaseModel):
    content: str
    role: str

class Choice(BaseModel):
    finish_reason: str
    index: int
    message: Message
    logprobs: Optional[float] = None

class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class OpenAIResponse(BaseModel):
    choices: List[Choice]
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    id: str
    model: str
    object: str = "chat.completion"
    usage: Usage

    class Config:
        allow_population_by_field_name = True

    @classmethod
    def create(cls, content: str, model: str, prompt_tokens: int = -1, completion_tokens: int = -1) -> 'OpenAIResponse':
        return cls(
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=content,
                        role="assistant"
                    )
                )
            ],
            id=f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            model=model,
            usage=Usage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    
    def to_dict(self) -> Dict:
        return {
            "choices": [
                {
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role
                    },
                    "logprobs": choice.logprobs
                }
                for choice in self.choices
            ],
            "created": self.created,
            "id": self.id,
            "model": self.model,
            "object": self.object,
            "usage": {
                "completion_tokens": self.usage.completion_tokens,
                "prompt_tokens": self.usage.prompt_tokens,
                "total_tokens": self.usage.total_tokens
            }
        }
    
    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)