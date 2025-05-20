"""
Unit tests for the chatbot functionality.
"""

import os
import pytest
from pathlib import Path
from src.chatbot import Chatbot
from src.data_loader import DataLoader
from src.rag_engine import RAGEngine

# Test data
SAMPLE_TEXT = """
This is a sample text for testing the chatbot.
It contains some information about artificial intelligence.
AI is a broad field of computer science focused on creating intelligent machines.
Machine learning is a subset of AI that focuses on training models using data.
"""

@pytest.fixture
def chatbot():
    """Create a chatbot instance for testing."""
    return Chatbot(
        model_name="gpt-3.5-turbo-0125",
        temperature=0.7,
        chunk_size=100,
        chunk_overlap=20,
    )

def test_chatbot_initialization(chatbot):
    """Test chatbot initialization."""
    assert chatbot is not None
    assert chatbot.data_loader is not None
    assert chatbot.rag_engine is not None
    assert isinstance(chatbot.conversation_history, list)
    assert len(chatbot.conversation_history) == 0

def test_process_text(chatbot):
    """Test processing text input."""
    chatbot.process_text(SAMPLE_TEXT)
    # Verify that the vector store was created
    assert chatbot.rag_engine.vector_store is not None

def test_ask_question(chatbot):
    """Test asking a question to the chatbot."""
    # First process some text
    chatbot.process_text(SAMPLE_TEXT)
    
    # Ask a question
    response = chatbot.ask("What is AI?")
    
    # Verify response structure
    assert isinstance(response, dict)
    assert "answer" in response
    assert "sources" in response
    assert isinstance(response["answer"], str)
    assert isinstance(response["sources"], list)

def test_conversation_history(chatbot):
    """Test conversation history management."""
    # Process text and ask a question
    chatbot.process_text(SAMPLE_TEXT)
    chatbot.ask("What is machine learning?")
    
    # Verify conversation history
    history = chatbot.get_conversation_history()
    assert len(history) == 1
    assert "question" in history[0]
    assert "answer" in history[0]
    assert "sources" in history[0]

def test_clear_conversation(chatbot):
    """Test clearing conversation history."""
    # Add some conversation
    chatbot.process_text(SAMPLE_TEXT)
    chatbot.ask("What is AI?")
    
    # Clear conversation
    chatbot.clear_conversation()
    
    # Verify history is cleared
    assert len(chatbot.get_conversation_history()) == 0

def test_export_conversation(chatbot, tmp_path):
    """Test exporting conversation to file."""
    # Add some conversation
    chatbot.process_text(SAMPLE_TEXT)
    chatbot.ask("What is machine learning?")
    
    # Export conversation
    export_path = tmp_path / "test_conversation.txt"
    chatbot.export_conversation(str(export_path))
    
    # Verify file was created
    assert export_path.exists()
    assert export_path.stat().st_size > 0

def test_invalid_document_path(chatbot):
    """Test handling of invalid document path."""
    with pytest.raises(FileNotFoundError):
        chatbot.load_documents("nonexistent_file.txt")

def test_unsupported_file_type(chatbot, tmp_path):
    """Test handling of unsupported file types."""
    # Create a file with unsupported extension
    test_file = tmp_path / "test.xyz"
    test_file.write_text("test content")
    
    with pytest.raises(ValueError):
        chatbot.load_documents(str(test_file))