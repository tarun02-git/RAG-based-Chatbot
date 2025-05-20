"""
Chatbot module implementing the chat interface using the RAG engine.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from datetime import datetime
from src.data_loader import DataLoader
from src.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chatbot:
    """Implements the chat interface using the RAG engine."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.7,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_directory: str = "./data/chroma_db",
    ):
        """
        Initialize the Chatbot.

        Args:
            model_name (str): Name of the OpenAI model to use
            temperature (float): Temperature for model generation
            chunk_size (int): Size of text chunks for splitting documents
            chunk_overlap (int): Overlap between chunks
            persist_directory (str): Directory to persist the vector store
        """
        self.data_loader = DataLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.rag_engine = RAGEngine(
            model_name=model_name,
            temperature=temperature,
            persist_directory=persist_directory,
        )
        self.conversation_history: List[Dict[str, Any]] = []
        logger.info("Successfully initialized Chatbot")

    def load_documents(self, path: str) -> None:
        """
        Load documents from a file or directory.

        Args:
            path (str): Path to the document or directory
        """
        try:
            path = Path(path)
            if path.is_file():
                documents = self.data_loader.load_document(str(path))
            elif path.is_dir():
                documents = self.data_loader.load_directory(str(path))
            else:
                raise ValueError(f"Invalid path: {path}")

            if documents:
                self.rag_engine.create_vector_store(documents)
                logger.info(f"Successfully loaded documents from {path}")
            else:
                logger.warning(f"No documents found at {path}")

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def process_text(self, text: str) -> None:
        """
        Process raw text and add it to the knowledge base.

        Args:
            text (str): Raw text to process
        """
        try:
            documents = self.data_loader.process_text(text)
            if documents:
                self.rag_engine.create_vector_store(documents)
                logger.info("Successfully processed text")
            else:
                logger.warning("No content extracted from text")

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot.

        Args:
            question (str): The question to ask

        Returns:
            Dict[str, Any]: Dictionary containing the answer and conversation history
        """
        try:
            # Check if the question is about the creator
            creator_keywords = ["who created you", "who is your creator", "your creator"]
            if any(keyword in question.lower() for keyword in creator_keywords):
                response = {"answer": "I was created by Tarun Agarwal.", "source_documents": []}
            else:
                # Get response from RAG engine
                response = self.rag_engine.query(question)

            # Create conversation entry
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": response["answer"],
                "sources": response["source_documents"],
            }

            # Update conversation history
            self.conversation_history.append(conversation_entry)

            logger.info(f"Successfully processed question: {question}")
            return conversation_entry

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.

        Returns:
            List[Dict[str, Any]]: List of conversation entries
        """
        return self.conversation_history

    def clear_conversation(self) -> None:
        """Clear the conversation history and memory."""
        self.conversation_history = []
        self.rag_engine.clear_memory()
        logger.info("Successfully cleared conversation history and memory")

    def export_conversation(self, file_path: str) -> None:
        """
        Export the conversation history to a file.

        Args:
            file_path (str): Path to save the conversation history
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                for entry in self.conversation_history:
                    f.write(f"Timestamp: {entry['timestamp']}\n")
                    f.write(f"Question: {entry['question']}\n")
                    f.write(f"Answer: {entry['answer']}\n")
                    f.write("Sources:\n")
                    for source in entry["sources"]:
                        f.write(f"- {source['content'][:200]}...\n")
                    f.write("\n" + "="*80 + "\n\n")

            logger.info(f"Successfully exported conversation to {file_path}")

        except Exception as e:
            logger.error(f"Error exporting conversation: {str(e)}")
            raise