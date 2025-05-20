"""
Data loader module for handling different document formats and preparing them for the RAG pipeline.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and processing of documents for the RAG pipeline."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        supported_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the DataLoader.

        Args:
            chunk_size (int): Size of text chunks for splitting documents
            chunk_overlap (int): Overlap between chunks
            supported_extensions (List[str]): List of supported file extensions
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions or [".pdf", ".docx", ".txt"]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on its file extension.

        Args:
            file_path (str): Path to the document

        Returns:
            List[Document]: List of processed document chunks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")

        try:
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif extension == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif extension == ".txt":
                loader = TextLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
            return self.text_splitter.split_documents(documents)

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path (str): Path to the directory containing documents

        Returns:
            List[Document]: List of processed document chunks from all files
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        all_documents = []
        for extension in self.supported_extensions:
            try:
                loader = DirectoryLoader(
                    str(directory_path),
                    glob=f"**/*{extension}",
                    loader_cls={
                        ".pdf": PyPDFLoader,
                        ".docx": Docx2txtLoader,
                        ".txt": TextLoader,
                    }[extension],
                )
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} documents with extension {extension}")
            except Exception as e:
                logger.warning(f"Error loading {extension} files: {str(e)}")

        if not all_documents:
            logger.warning(f"No documents found in {directory_path}")
            return []

        return self.text_splitter.split_documents(all_documents)

    def process_text(self, text: str) -> List[Document]:
        """
        Process raw text into document chunks.

        Args:
            text (str): Raw text to process

        Returns:
            List[Document]: List of processed text chunks
        """
        try:
            document = Document(page_content=text)
            return self.text_splitter.split_documents([document])
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise