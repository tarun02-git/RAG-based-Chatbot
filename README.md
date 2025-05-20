# RAG-based-Chatbot
## TurboTT - RAG Chatbot

###Created by: Tarun Agarwal

A professional Retrieval-Augmented Generation (RAG) Chatbot built with LangChain and Streamlit. This chatbot uses external data sources to provide accurate and contextual responses to user queries.

## Features

- RAG-based question answering using LangChain
- Support for multiple document formats (PDF, DOCX, TXT)
- Vector-based semantic search using ChromaDB
- Interactive web interface using Streamlit
- Comprehensive documentation and testing
- Environment variable management for API keys

## Project Structure

```
.
├── data/                   # Data directory for knowledge base
├── src/                    # Source code
│   ├── data_loader.py     # Data loading and processing
│   ├── rag_engine.py      # RAG pipeline implementation
│   ├── chatbot.py         # Chatbot core logic
│   └── utils.py           # Utility functions
├── tests/                  # Unit tests
├── requirements.txt        # Project dependencies
├── .env.example           # Example environment variables
└── README.md              # Project documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
```
Then add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload your documents or use the pre-loaded dataset

4. Start chatting with the bot!

## Development

- Code formatting: `black src/ tests/`
- Linting: `flake8 src/ tests/`
- Running tests: `pytest tests/`

## Testing

The project includes comprehensive unit tests. Run them using:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG framework
- OpenAI for the language models
- Streamlit for the web interface
