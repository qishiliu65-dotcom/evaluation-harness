class MockSystem:
    def __init__(self, response: str):
        self.response = response

    def generate(self, input_text: str) -> str:
        return self.response