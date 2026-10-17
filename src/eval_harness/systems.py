from typing import Protocol


class SystemUnderTest(Protocol):
    def generate(self, input_text: str) -> str:
        ...


class MockSystem:
    def __init__(self, response: str):
        self.response = response

    def generate(self, input_text: str) -> str:
        return self.response