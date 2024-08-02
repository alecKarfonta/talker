from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json

class GenerateRequest(BaseModel):
    prompts: list = Field(default=[], description="Input prompt for text generation")
    max_new_tokens: int = Field(default=100, description="Maximum number of new tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    top_k: int = Field(default=40, description="Top-k sampling parameter")


    def to_dict(self) -> Dict:
        return {
            "prompts": self.prompts,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)