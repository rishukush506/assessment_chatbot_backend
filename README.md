# Financial Assessment Chatbot

An AI-powered conversational assessment tool that evaluates users' financial behavior traits through natural conversation. The system uses LangGraph workflows to conduct intelligent assessments and generate personalized financial personas.

## Project Structure

```
assessment_server/
├── financial_assessment.py    # Core assessment logic and LangGraph workflow
├── server.py                 # FastAPI web server
├── test.py                   # Test utilities
├── pyproject.toml           # Project dependencies and configuration
├── uv.lock                  # Dependency lock file
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for:
  - OpenAI (required)
  - Google Gemini (optional, for fallbacks)

## Installation

### 1. Setup Project

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` file with your API keys:

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Gemini API Keys (for fallback)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY_1=your_additional_gemini_key_1
GEMINI_API_KEY_2=your_additional_gemini_key_2
# ... add more as needed (up to GEMINI_API_KEY_10)
```

## Usage

### Starting the Server

```bash
# Using uv
uv run python server.py

The server will start on `http://localhost:8000`

### API Documentation

Once the server is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### Core Endpoints

**POST `/chat`** - Main conversation endpoint

```json
{
  "message": "I usually save about 20% of my income",
  "state": {...}  // Optional: previous conversation state
}
```

**GET `/empty-state`** - Get fresh conversation state

```json
{
  "current_priority": "awareness",
  "current_iteration": -2,
  "messages": [],
  ...
}
```

**POST `/persona`** - Generate financial persona summary

```json
{
  "state": {...}  // Completed assessment state
}
```

#### Utility Endpoints

**GET `/`** - Health check
**GET `/health`** - Detailed health status
**POST `/process-state`** - State validation and formatting

### Example Conversation Flow

1. **Start**: Get empty state from `/empty-state`
2. **Chat**: Send messages to `/chat` with the current state
3. **Continue**: Use returned state for subsequent messages
4. **Complete**: When assessment is finished, call `/persona` for summary

## Development

### Adding New Dependencies

```bash
# Add a new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update dependencies
uv sync
```

### Code Structure

- **`financial_assessment.py`**: Contains the core assessment logic, LangGraph workflow, and trait evaluation functions
- **`server.py`**: FastAPI application with API endpoints and request/response handling
- **State Management**: Uses TypedDict for structured state tracking across conversation turns

### Configuration

Key configuration constants in `financial_assessment.py`:

- `MAX_QUESTIONS`: Maximum questions per trait (default: 2) [Keep in mind that the question iteration starts with 0 for most raits and -2 for awareness]
- `CONFIDENCE_THRESHOLD`: Minimum confidence to advance traits (default: 7)
- `TRAITS`: List of assessed financial traits

### Common Issues

**Import Errors**: Ensure virtual environment is activated and dependencies are installed

```bash
uv sync
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

**API Key Errors**: Verify `.env` file exists and contains valid API keys

**Port Conflicts**: Change the port in `server.py` if 8000 is occupied

**Model Fallback Issues**: Ensure at least OpenAI API key is configured

## Features

- **Conversational Assessment**: Natural dialogue-based evaluation of financial traits
- **Multi-Trait Analysis**: Assesses 6 key financial behavior dimensions:
  - Financial Awareness
  - Self Control
  - Preparedness
  - Information Seeking
  - Risk Seeking
  - Reaction to External Events
- **Adaptive Questioning**: Switches between conversation-based and option-based assessment methods
- **AI Model Fallbacks**: Uses OpenAI GPT-4 with Google Gemini fallbacks for reliability
- **RESTful API**: FastAPI-based server with CORS support
- **Persona Generation**: Creates comprehensive financial behavior summaries