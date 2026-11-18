from .models import Document

def load_text_document(path: str, document_id: str,) -> Document:
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    return Document(id=document_id, text=text)