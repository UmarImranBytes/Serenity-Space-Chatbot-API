# Serenity Space Chatbot API

A compassionate, conversational AI assistant API built with FastAPI and powered by Google's Gemini 1.5 Pro. SerenityBot is designed to provide emotional wellness support by analyzing user input for mood, tone, and context, and delivering empathetic, actionable mental health strategies.

## Features

- Mood & Tone Analysis: Automatically detects the user's emotional state (e.g., happy, sad, anxious, stressed) and conversational tone.
- Context-Aware Responses: Tailors advice based on the user's specific context (e.g., work stress, exam pressure, relationship issues).
- Continuous Conversations: Maintains chat history using native Gemini Chat Sessions for natural, flowing follow-up interactions.
- Structured Coping Strategies: Provides actionable, step-by-step guidance including grounding exercises, mindfulness, and CBT techniques.
- Special Pathways: Quick-access conversational routes for specific needs (e.g., typing "mindfulness", "tools", "plan", "resources").

## Tech Stack

- Framework: FastAPI
- AI Model: Google Generative AI (Gemini 1.5 Pro)
- Data Validation: Pydantic
- Language: Python 3.8+

## Getting Started

### 1. Clone the repository
git clone https://github.com/UmarImranBytes/Serenity-Space-Chatbot-API.git
cd Serenity-Space-Chatbot-API

### 2. Set up a virtual environment (Optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### 3. Install dependencies
Make sure you have a requirements.txt file, or install the packages manually:
pip install fastapi uvicorn google-generativeai python-dotenv pydantic

### 4. Configure Environment Variables
Create a .env file in the root directory and add your Google Gemini API key:
GEMINI_API_KEY=your_gemini_api_key_here

### 5. Run the API Server
uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000. You can view the interactive Swagger documentation at http://127.0.0.1:8000/docs.

## API Endpoints

### GET /
Returns a welcome message and verifies the API is running.

### POST /generate_response
Generates an empathetic response based on user input.

Request Body Example:
{
  "mood": "anxious",
  "reason": "upcoming exams",
  "input_text": "I can't stop worrying about failing.",
  "conversation_id": "conv_12345" 
}
(Note: Omit conversation_id on the first request to start a new chat session).

Response Example:
{
  "user_id": "user_abc123",
  "input": {
    "mood": "anxious",
    "age": null,
    "reason": "upcoming exams",
    "input_text": "I can't stop worrying about failing."
  },
  "response": "I'm really sorry you're feeling this way... [Detailed response steps]",
  "conversation_id": "conv_12345",
  "timestamp": "2026-04-24 12:00:00"
}

## Author
Umar
- GitHub: @UmarImranBytes (https://github.com/UmarImranBytes)
- Email: iumar4770@gmail.com
