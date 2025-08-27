import requests
import json

# Base URL of your FastAPI server
BASE_URL = "http://localhost:8000"


def test_chat_api():
    """Test the chat endpoint"""

    # Start with empty state
    response = requests.get(f"{BASE_URL}/empty-state")
    current_state = response.json()

    # Send first message
    chat_data = {
        "message": "Hello, I'd like to get a financial assessment.",
        "state": current_state
    }

    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    result = response.json()

    print("Bot Response:", result["response"])
    current_state = result["updated_state"]

    # Continue conversation
    chat_data = {
        "message": "I usually save about 20% of my income each month.",
        "state": current_state
    }

    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    result = response.json()

    print("Bot Response:", result["response"])

    current_state = result["updated_state"]

    # Continue conversation
    persona_data = {
        "state": current_state
    }

    response = requests.post(f"{BASE_URL}/persona", json=persona_data)
    result = response.json()

    print("Bot Response:", result)
    # print("Current State Keys:", list(result["updated_state"].keys()))


def test_state_processing():
    """Test the process-state endpoint"""

    # Get empty state
    response = requests.get(f"{BASE_URL}/empty-state")
    state = response.json()

    # Add a message to the state
    state["messages"].append({
        "type": "human",
        "content": "I'm interested in learning about my financial habits."
    })

    # Process the state
    response = requests.post(
        f"{BASE_URL}/process-state", json={"state": state})
    result = response.json()

    print("Updated State Status:", result["status"])
    print("Number of messages:", len(result["updated_state"]["messages"]))


if __name__ == "__main__":
    test_chat_api()
    print("\n" + "="*10 + "\n")
    test_state_processing()
