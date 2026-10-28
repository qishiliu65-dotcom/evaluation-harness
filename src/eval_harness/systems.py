from typing import Protocol
from openai import OpenAI
from .models import AnswerOutput
import os


class SystemUnderTest(Protocol):
    def generate(self, input_text: str) -> str:
        ...

class StructuredSystemUnderTest(Protocol):
    def generate_structured(self, input_text: str) -> AnswerOutput:
        ...


class MockSystem:
    def __init__(self, response: str):
        self.response = response

    def generate(self, input_text: str) -> str:
        return self.response


class LLMSystem:
    def __init__(self, model: str):
        self.model = model

        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_BASE_URL")

        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")

        if not base_url:
            raise ValueError("DASHSCOPE_BASE_URL is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, input_text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("LLM returned no text content")
      
        return content

    def generate_structured(self, input_text: str) -> AnswerOutput:
        response = self.client.chat.completions.parse(
        model=self.model,
        messages=[
            {
                "role": "user",
                "content": input_text
            }
        ],
        response_format=AnswerOutput
        )

        # 从模型返回结果里，取出“已经被 SDK 解析成 Python 对象”的结构化结果，然后保存到变量 parsed
        parsed = response.choices[0].message.parsed

        if not parsed:
            raise ValueError("LLM returned no structured output")

        return parsed
