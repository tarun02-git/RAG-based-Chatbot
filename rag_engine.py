"""
RAG Engine module implementing the core Retrieval-Augmented Generation pipeline.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """Implements the core RAG pipeline for question answering."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.7,
        persist_directory: str = "./data/chroma_db",
    ):
        """
        Initialize the RAG Engine.

        Args:
            model_name (str): Name of the OpenAI model to use
            temperature (float): Temperature for model generation
            persist_directory (str): Directory to persist the vector store
        """
        self.model_name = model_name
        self.temperature = temperature
        self.persist_directory = persist_directory
        self.vector_store = None
        self.qa_chain = None
        self.memory = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize the LLM, embeddings, and memory components."""
        try:
            # Initialize OpenAI components
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
            )
            self.embeddings = OpenAIEmbeddings()

            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )

            logger.info("Successfully initialized RAG components")
        except Exception as e:
            logger.error(f"Error initializing RAG components: {str(e)}")
            raise

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create or update the vector store with documents.

        Args:
            documents (List[Document]): List of documents to add to the vector store
        """
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")

            # Create or load vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

            # Create QA chain
            self._create_qa_chain()

            logger.info(f"Successfully created vector store with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def _create_qa_chain(self) -> None:
        """Create the question-answering chain with custom prompt."""
        try:
            # Custom prompt template
            template = """You are an AI assistant that helps answer questions based on the provided context.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer concise.

            Context: {context}

            Chat History: {chat_history}
            Human: {question}
            Assistant:"""

            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=template,
            )

            # Create the QA chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True,
            )

            logger.info("Successfully created QA chain")
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.

        Args:
            question (str): The question to ask

        Returns:
            Dict[str, Any]: Dictionary containing the answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please create vector store first.")

        try:
            # Get response from QA chain
            response = self.qa_chain({"question": question})

            # Extract relevant information
            result = {
                "answer": response["answer"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in response.get("source_documents", [])
                ],
            }

            logger.info(f"Successfully processed question: {question}")
            return result

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise

    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("Successfully cleared conversation memory")