# FastAPI server for the Financial Assessment Chatbot
from fastapi import FastAPI, HTTPException , BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from pymongo import MongoClient, ReturnDocument
from datetime import datetime
import uuid
import dotenv
import json
import os



# Import the financial assessment components
from financial_assessment import assessment_graph, EMPTY_STATE, generate_persona

app = FastAPI(title="Financial Assessment Chatbot API", version="1.0.0")


dotenv.load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI,tls=True, tlsAllowInvalidCertificates=True)

db = client["chatbot_db"]
messages = db["messages_user_testing"]
counters = db["counters_user_testing"]
persona_collection=db["persona_user_testing"]

def generate_user_id():
    counter = counters.find_one_and_update(
        {"_id": "user_id"},                   # document with key user_id
        {"$inc": {"seq": 1}},                 # increment by 1
        upsert=True,                          # create if missing
        return_document=ReturnDocument.AFTER  # return updated document
    )
    return f"user{counter['seq']}"

# sessions: Dict[str, List[Dict]] = {}

@app.post("/start-session")
async def start_new_session():
    user_id = generate_user_id()
    session_id = str(uuid.uuid4())  # unique per chat refresh
    print(user_id)
    print(session_id)
    # sessions[session_id] = []
    print("👉 Frontend reloaded and requested a new session")
    return {"user_id":user_id , "session_id": session_id}
 

def save_message(user_id,session_id, user_res, ai_res,current_priority,llm_confidence,parameter_score, parameter_rationale):

    try:
        messages.insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "user_response":user_res,
            "ai_response":ai_res,

            "current_priority":current_priority,
            
            "parameter_score":json.dumps(parameter_score),
            "llm_confidence":json.dumps(llm_confidence),
            "parameter_rationale":json.dumps(parameter_rationale),

            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print("Message Saved")
    except Exception as e:
        # print(e)
        raise HTTPException(
            status_code=500, detail=f"Failed to save in database: {str(e)}")


def save_persona(user_id,session_id,persona,llm_confidence,parameter_score, parameter_rationale):

    try:
        persona_collection.insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "persona":persona,
            "parameter_score":json.dumps(parameter_score),
            "llm_confidence":json.dumps(llm_confidence),
            "parameter_rationale":json.dumps(parameter_rationale),

            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print("Persona Saved")
    except Exception as e:
        # print(e)
        raise HTTPException(
            status_code=500, detail=f"Failed to save in database: {str(e)}")


# SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
# FROM_EMAIL = os.getenv("FROM_EMAIL")

# def send_mail(to_email: str, subject: str, body: str):
#     """Send email using SendGrid API"""
#     try:
#         message = Mail(
#             from_email=FROM_EMAIL,
#             to_emails=to_email,
#             subject=subject,
#             html_content=body   # supports HTML content
#         )
#         sg = SendGridAPIClient(SENDGRID_API_KEY)
#         response = sg.send(message)
#         print(f"✅ Mail sent to {to_email}, status: {response.status_code}")
#     except Exception as e:
#         print(f"❌ Error: {e}")


origins = [
    "http://localhost:3000",  # local dev
    "https://assessment-chatbot-nu.vercel.app/"  # your deployed vercel frontend URL
]

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models


class MessageModel(BaseModel):
    type: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str
    user_id: str
    session_id: str
    state: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    response: str
    updated_state: Dict[str, Any]
    status: str


class StateRequest(BaseModel):
    """Request model for state operations"""
    user_id:str
    session_id: str
    state: Dict[str, Any]


class StateResponse(BaseModel):
    """Response model for state operations"""
    updated_state: Dict[str, Any]
    status: str


# Message serialization utilities

def serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain messages to JSON-serializable format"""
    serialized = []
    for msg in messages:
        serialized.append({
            "type": msg.type,
            "content": msg.content
        })
    return serialized


def deserialize_messages(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """Convert JSON messages back to LangChain message objects"""
    deserialized = []
    for msg in messages:
        if msg["type"] == "human":
            deserialized.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            deserialized.append(AIMessage(content=msg["content"]))
    return deserialized


def prepare_state_for_serialization(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare state for JSON serialization by converting messages"""
    serialized_state = state.copy()
    if "messages" in serialized_state:
        serialized_state["messages"] = serialize_messages(
            serialized_state["messages"])
    return serialized_state


def prepare_state_for_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare state for graph execution by converting messages back to objects"""
    prepared_state = state.copy()
    if "messages" in prepared_state:
        prepared_state["messages"] = deserialize_messages(
            prepared_state["messages"])
    return prepared_state


# API Endpoints



@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Financial Assessment Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user messages through the assessment graph
    """
    try:

        # Use provided state or start fresh
        current_state = request.state if request.state else EMPTY_STATE.copy()

        # Terminating process if user writes end, quit
        if request.message.lower().strip() in ["end","quit"]:
            return ChatResponse(
                response="terminate",
                updated_state=current_state,
                status="success"
            ) 

        # Convert serialized messages back to LangChain objects
        current_state = prepare_state_for_graph(current_state)

        # Add user message to conversation
        current_state["messages"].append(HumanMessage(content=request.message))

        # Process through the assessment workflow
        updated_state = assessment_graph.invoke(current_state)

        # Extract the AI's response from the last message
        ai_messages = [
            msg for msg in updated_state["messages"] if msg.type == "ai"]
        response_content = ai_messages[-1].content if ai_messages else "I'm sorry, I couldn't generate a response."

        # print(request.json())
        user_id=request.user_id
        session_id=request.session_id

        # Convert back to serializable format
        serialized_state = prepare_state_for_serialization(updated_state)
        print("serialized_state")
        print(serialized_state)

        # Saving data
        print(user_id)  ## User Id for database
        print(session_id) 
        print(current_state["messages"][-1].content)  ## User message 
        print(ai_messages[-1].content)   ## System Message
        

        current_priority=serialized_state["current_priority"]

        score={
            "self_control_score"  : serialized_state['self_control_score'],
            "preparedness_score "  : serialized_state['preparedness_score'],
            "information_seeking_score ":serialized_state['information_seeking_score'],
            "risk_seeking_score "   :serialized_state['risk_seeking_score'],
            "awareness_score  "  :serialized_state['awareness_score'],
            "reaction_to_external_events_score "  : serialized_state['reaction_to_external_events_score']
        }

        confidence={
            "self_control_confidence  ":  serialized_state['self_control_confidence'],
            "preparedness_confidence   ": serialized_state['preparedness_confidence'],
            "information_seeking_confidence ": serialized_state['information_seeking_confidence'],
            "risk_seeking_confidence    ":serialized_state['risk_seeking_confidence'],
            "awareness_confidence    ":serialized_state['awareness_confidence'],
            "reaction_to_external_events_confidence   ": serialized_state['reaction_to_external_events_confidence']
        }

        rationale={
            "self_control_rationale  ":  serialized_state['self_control_rationale'],
            "preparedness_rationale   ": serialized_state['preparedness_rationale'],
            "information_seeking_rationale ": serialized_state['information_seeking_rationale'],
            "risk_seeking_rationale    ":serialized_state['risk_seeking_rationale'],
            "awareness_rationale    ":serialized_state['awareness_rationale'],
            "reaction_to_external_events_rationale   ": serialized_state['reaction_to_external_events_rationale']
        }

        #  Function for saving data in database

        if current_priority!='persona':
            save_message(
                user_id=user_id,
                session_id=session_id,
                user_res=current_state["messages"][-1].content,
                ai_res=ai_messages[-1].content,
                current_priority=current_priority,
                parameter_score=score,
                llm_confidence=confidence,
                parameter_rationale=rationale
            )
        else:
            save_persona(
            user_id=user_id,
            session_id=session_id,
            persona=response_content,
            llm_confidence=confidence,
            parameter_score=score,
            parameter_rationale=rationale
        )

        return ChatResponse(
            response=response_content,
            updated_state=serialized_state,
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat: {str(e)}"
            )


@app.post("/process-state", response_model=StateResponse)
async def process_state(request: StateRequest):
    """
    Process a state object (mainly for state validation/formatting)
    """
    try:
        # Prepare state for graph execution
        current_state = prepare_state_for_graph(request.state)

        # Convert back to serializable format
        serialized_state = prepare_state_for_serialization(current_state)
        print("serialized_state")
        print(serialized_state)
        return StateResponse(
            updated_state=serialized_state,
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing state: {str(e)}")


@app.get("/empty-state")
async def get_empty_state():
    """
    Get a fresh state to start a new assessment session
    """
    return prepare_state_for_serialization(EMPTY_STATE.copy())


@app.post("/persona")
async def get_persona_endpoint(request: StateRequest):
    """ 
    Generate final persona summary based on completed assessment
    """
    try:
        if not request.state:
            raise HTTPException(
                status_code=400, detail="State is required to generate persona")

        current_state = request.state
        print("Current State:", current_state)

        user_id = request.user_id
        session_id=request.session_id
        print(user_id)
        print(session_id)

        # Prepare state for persona generation
        current_state = prepare_state_for_graph(current_state)

        # Generate the financial persona summary
        response_content = generate_persona(current_state)
        print("response_content")
        print(response_content)



        # send_mail(
        #     to_email="rishukush506@gmail.com", 
        #     subject="User id for Chatbot Testing", 
        #     body=user_id,
        # )
        score={
            "self_control_score"  : current_state['self_control_score'],
            "preparedness_score "  : current_state['preparedness_score'],
            "information_seeking_score ":current_state['information_seeking_score'],
            "risk_seeking_score "   :current_state['risk_seeking_score'],
            "awareness_score  "  :current_state['awareness_score'],
            "reaction_to_external_events_score "  : current_state['reaction_to_external_events_score']
        }

        confidence={
            "self_control_confidence  ":  current_state['self_control_confidence'],
            "preparedness_confidence   ": current_state['preparedness_confidence'],
            "information_seeking_confidence ": current_state['information_seeking_confidence'],
            "risk_seeking_confidence    ":current_state['risk_seeking_confidence'],
            "awareness_confidence    ":current_state['awareness_confidence'],
            "reaction_to_external_events_confidence   ": current_state['reaction_to_external_events_confidence']
        }

        rationale={
            "self_control_rationale  ":  current_state['self_control_rationale'],
            "preparedness_rationale   ": current_state['preparedness_rationale'],
            "information_seeking_rationale ": current_state['information_seeking_rationale'],
            "risk_seeking_rationale    ":current_state['risk_seeking_rationale'],
            "awareness_rationale    ":current_state['awareness_rationale'],
            "reaction_to_external_events_rationale   ": current_state['reaction_to_external_events_rationale']
        }

        save_persona(
            user_id=user_id,
            session_id=session_id,
            persona=response_content,
            llm_confidence=confidence,
            parameter_score=score,
            parameter_rationale=rationale
        )

        return {
            "response": response_content,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing persona: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint
    """
    return {"status": "healthy", "service": "Financial Assessment Chatbot"}

# Server startup configuration
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
