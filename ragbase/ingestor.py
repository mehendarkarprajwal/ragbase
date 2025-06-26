from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from ragbase.config import Config

from concurrent.futures import ThreadPoolExecutor


class Ingestor:
    def __init__(self, max_workers: int = 16):
        self.embeddings = FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)
        # self.embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

        self.semantic_splitter = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="interquartile"
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            add_start_index=True,
        )
        self.max_workers = max_workers

    def process_document(self, doc_path: Path):
        """Load, split, and process a single document."""
        # loaded_documents = PyPDFium2Loader(str(doc_path)).load()
        if str(doc_path).endswith(".pdf"):
            loaded_documents = PyPDFium2Loader(str(doc_path)).load()
        elif str(doc_path).endswith(".txt"):
            loaded_documents = TextLoader(str(doc_path)).load()
        else:
            raise ValueError("Unsupported file type")
        document_text = "\n".join([doc.page_content for doc in loaded_documents])
        # return self.recursive_splitter.split_documents(
        #     self.semantic_splitter.create_documents([document_text])
        # )
        return self.recursive_splitter.split_documents(
            self.semantic_splitter.create_documents([document_text])
        )

    def ingest(self, doc_paths: List[Path]) -> Qdrant:
        """Efficiently process and ingest multiple documents into the vector store."""
        documents = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.process_document, doc_paths)
            for doc_list in results:
                documents.extend(doc_list)
        
        return Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            path=Config.Path.DATABASE_DIR,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )