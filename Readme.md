# DataSense - Conversational AI Assistant

## 📋 Project Overview
DataSense is a conversational AI assistant with a friendly, data-enthusiast personality. It provides three core services as required by Assignment 2, all integrated into a Gradio chat interface.

## 📁 Project Structure (Simplified)

05_src/assignment_chat/
├── init.py # Package initialization
├── main.py # Main application (all services + chat logic)
├── app.py # Gradio interface launcher
└── README.md # This file


## 🎯 Services Implemented

### Service 1: API Service - Joke Transformer
- **Source**: Uses the free [icanhazdadjoke.com](https://icanhazdadjoke.com/) public API
- **Transformation**: Fetches random dad jokes and transforms them into data science humor using GPT-4o-mini
- **Output**: Returns rewritten jokes, not verbatim API output

### Service 2: Semantic Query Service - Music Review Search
- **Dataset**: Sample music reviews (5 reviews across different genres)
- **Vector DB**: ChromaDB with persistent client (`./chroma_data`)
- **Embeddings**: OpenAI text-embedding-3-small model
- **Search**: Semantic search through review content
- **Data Handling**: Uses pandas for data manipulation (as allowed)

### Service 3: Function Calling - Weather Information
- **Implementation**: OpenAI function calling
- **Function**: `get_weather` with city parameter
- **Processing**: Extracts location from natural language queries

## 🧠 Personality Design
DataSense speaks in a warm, professional tone and uses data analogies. 
Example: "Think of embeddings as a GPS for words."

## 💾 Memory Management
- Maintains conversation history (last 20 messages)
- Context window management (keeps last 10 messages + system prompt)
- Simple memory management when conversations get long

## 🛡️ Guardrails Implemented
1. **System Prompt Protection**: Prevents revealing or modifying system prompt
2. **Restricted Topics**: Blocks discussions about:
   - Cats and dogs
   - Horoscopes and zodiac signs
   - Taylor Swift
3. **Polite Redirection**: Provides alternative suggestions when topics are blocked

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- API Gateway key from the course

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd 05_src/assignment_chat