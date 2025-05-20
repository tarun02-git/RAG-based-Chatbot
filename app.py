"""
Streamlit web application for the RAG-based chatbot.
"""

import os
import streamlit as st
from pathlib import Path
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from src.chatbot import Chatbot

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="TurboTT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message .avatar {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 0.5rem;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .source-docs {
        font-size: 0.8rem;
        color: #a0a0a0;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

def display_chat_message(message, is_user=False):
    """Display a chat message with proper styling."""
    avatar = "👤" if is_user else "🤖"
    message_class = "user" if is_user else "assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="content">
            <div class="avatar">{avatar}</div>
            <div class="message">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_source_documents(sources):
    """Display source documents in a collapsible section."""
    if sources:
        with st.expander("View Source Documents"):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**Source {i}:**")
                st.text(source["content"][:500] + "...")
                st.markdown("---")

def main():
    """Main application function."""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("📚 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        # Load document
                        st.session_state.chatbot.load_documents(tmp_path)
                        st.session_state.uploaded_files.add(uploaded_file.name)
                        st.success(f"Successfully processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_path)

        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.chatbot.clear_conversation()
            st.session_state.messages = []
            st.success("Conversation cleared!")

        # Export conversation button
        if st.button("Export Conversation"):
            if st.session_state.messages:
                export_path = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                try:
                    st.session_state.chatbot.export_conversation(export_path)
                    st.success(f"Conversation exported to {export_path}")
                except Exception as e:
                    st.error(f"Error exporting conversation: {str(e)}")
            else:
                st.warning("No conversation to export")

    # Main chat interface
    st.title("🤖 TurboTT")
    st.markdown("""
    Welcome to TurboTT! Upload your documents in the sidebar and start asking questions.
    The chatbot will use the uploaded documents to provide accurate and contextual answers.
    """)

    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["content"], message["is_user"])
        if not message["is_user"] and "sources" in message:
            display_source_documents(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat
        st.session_state.messages.append({"content": prompt, "is_user": True})
        display_chat_message(prompt, is_user=True)

        # Get chatbot response
        try:
            response = st.session_state.chatbot.ask(prompt)
            
            # Add assistant message to chat
            st.session_state.messages.append({
                "content": response["answer"],
                "is_user": False,
                "sources": response["sources"],
            })
            
            # Display response
            display_chat_message(response["answer"])
            display_source_documents(response["sources"])

        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.session_state.messages.append({"content": error_message, "is_user": False})
            display_chat_message(error_message)

if __name__ == "__main__":
    main()