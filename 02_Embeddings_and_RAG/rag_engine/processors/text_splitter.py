from typing import List

class TextSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 256):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
        return chunks