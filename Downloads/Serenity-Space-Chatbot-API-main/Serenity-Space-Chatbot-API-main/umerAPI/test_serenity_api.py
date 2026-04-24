import requests
import pytest
from multiprocessing import Process
import uvicorn
import time
import os

# Add the parent directory to the path to allow imports from umerAPI
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from umerAPI.serenity_api import app

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

@pytest.fixture(scope="session", autouse=True)
def server():
    proc = Process(target=run_server, args=(), daemon=True)
    proc.start()
    time.sleep(5)  # Give the server some time to start
    yield
    proc.terminate()

def test_generate_response():
    """Test the /generate_response endpoint."""
    response = requests.post("http://localhost:8000/generate_response", json={"mood": "happy"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "conversation_id" in data
    assert "user_id" in data

def test_conversation_flow():
    """Test the conversation flow."""
    # Start a new conversation
    response = requests.post("http://localhost:8000/generate_response", json={"mood": "sad", "reason": "I had a bad day"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "conversation_id" in data
    conversation_id = data["conversation_id"]

    # Continue the conversation
    response = requests.post("http://localhost:8000/generate_response", json={"mood": "sad", "conversation_id": conversation_id, "input_text": "tell me more"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["conversation_id"] == conversation_id

    # Stop the conversation
    response = requests.post("http://localhost:8000/generate_response", json={"mood": "sad", "conversation_id": conversation_id, "stop": True})
    assert response.status_code == 200
    data = response.json()
    assert "Thank you for talking with me" in data["response"]
    assert data["conversation_id"] is None

def test_dynamic_conversational_flow():
    """Test that arbitrary user inputs correctly generate a dynamic Gemini acknowledgement."""
    response = requests.post("http://localhost:8000/generate_response", json={"mood": "stressed", "reason": "too much work"})
    assert response.status_code == 200
    data = response.json()
    conversation_id = data["conversation_id"]
    
    # Send a non-special keyword reply to test the Gemini acknowledgement fallback
    response = requests.post("http://localhost:8000/generate_response", json={
        "mood": "stressed",
        "conversation_id": conversation_id,
        "input_text": "I really don't think I have time to do any of those steps right now."
    })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 10 # Check that we got a substantial response
    assert data["conversation_id"] == conversation_id

def test_new_emotions():
    """Test one of the newly added emotions to ensure it is fetched from the mood library correctly."""
    response = requests.post("http://localhost:8000/generate_response", json={"mood": "frustrated", "reason": "code isn't working"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "Step" in data["response"] # Indicates steps were generated
    assert data["conversation_id"] is not None
