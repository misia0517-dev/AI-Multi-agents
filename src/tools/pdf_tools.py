from __future__ import annotations
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


@dataclass
class PdfRag:
    persist_dir: str = "chroma_db"
    collection_name: str = "business_context"

    def build(self, pdf_path: str) -> Chroma:
        # Support both .pdf and .md/.txt files
        if pdf_path.lower().endswith(".pdf"):
            docs = PyPDFLoader(pdf_path).load()
        else:
            docs = TextLoader(pdf_path, encoding="utf-8").load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        # Free local embeddings — no API key needed
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_dir,
        )

        # Workshop-friendly: always add chunks (wipe chroma_db if you want clean runs)
        vectordb.add_documents(chunks)
        return vectordb

    def retriever(self, vectordb: Chroma, k: int = 6):
        return vectordb.as_retriever(search_kwargs={"k": k})
