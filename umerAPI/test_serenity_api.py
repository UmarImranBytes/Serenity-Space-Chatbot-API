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
