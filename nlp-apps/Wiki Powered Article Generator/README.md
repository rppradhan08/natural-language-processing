# Wikipedia Powered Article Generator

The "Wikipedia Powered Article Generator" project aims to develop an intelligent system that automatically generates high-quality articles using information from Wikipedia. By leveraging natural language processing and machine learning techniques, this project streamlines content creation, saving time and effort. Users can customize topics and article lengths, while the system ensures accuracy, reliability, and adherence to ethical guidelines. This project revolutionizes content creation for domains like journalism, education, and research, unlocking the vast knowledge within Wikipedia for tailored article generation.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Below mentioned are the list of packages and their purpose:

1. `Streamlit` - Used to build the Web-based Application
2. `Langchain` - Used to build the LLM-based backend workflow
3. `OpenAI` - SDK to interact with OPENAI based models
4. `ChromaDB` - Vector storage to maintain Knowledgebase
5. `TikToken` - Tokenizer for OpenAI based LLMs

### Installing

Install following python packages to setup the environment.

```bash
    pip install streamlit==1.0.0 langchain openai wikipedia chromadb tiktoken
```

## Running the tests

Execute below command to test this application on localhost environment.

```bash
streamlit run app.py
```

## Built With

- **Python**: The core programming language used for developing the project.
- **Langchain**: Framework used to interact with OpenAI based LLM.
- **Streamlit**: Framework used to build the Webbased application.
