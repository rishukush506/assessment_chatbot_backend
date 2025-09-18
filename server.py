# FastAPI server for the Financial Assessment Chatbot
from fastapi import FastAPI, HTTPException , BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from pymongo import MongoClient, ReturnDocument
from datetime import datetime
from zoneinfo import ZoneInfo
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
# Production changes
messages = db["messages_user_testing"]
counters = db["counters_user_testing"]
persona_collection=db["persona_user_testing"]
feedback_collection = db["feedback_user_testing"]
# messages = db["messages"]
# counters = db["counters"]
# persona_collection=db["persona"]
# feedback_collection = db["feedback"]

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


class Feedback(BaseModel):
    responses: dict
    user_id: str
    session_id: str


#  Saving Feedback in database
def save_feedback_db(feedback: Feedback):
    try:
        feedback_collection.insert_one({
            "user_id": feedback.user_id,
            "session_id": feedback.session_id,
            "responses": feedback.responses,
            "timestamp": datetime.now(tz=ZoneInfo('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
        })
        print("Feedback Saved")
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Failed to save feedback in database: {str(e)}")

# Submit feedback endpoint
@app.post("/feedback")
async def save_feedback(feedback: Feedback):
    # For now just print, later save to DB
    print("User ID:", feedback.user_id)
    print("Session ID:", feedback.session_id)
    print("Responses:", feedback.responses)
    save_feedback_db(feedback)
    return {"status": "success", "message": "Feedback received"}
 

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

            "timestamp": datetime.now(tz=ZoneInfo('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
        })
        print("Message Saved")
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Failed to save in database: {str(e)}")


def save_persona(user_id,session_id,persona_label,avg_score,persona,llm_confidence,parameter_score, parameter_rationale):
    try:

        print("persona_label")
        print(persona_label)
        persona_collection.insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "persona":persona,
            "persona_label":persona_label,
            "average_score":json.dumps(avg_score),
            "parameter_score":json.dumps(parameter_score),
            "llm_confidence":json.dumps(llm_confidence),
            "parameter_rationale":json.dumps(parameter_rationale),
            "timestamp": datetime.now(tz=ZoneInfo('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
        })
        print("Persona Saved")
    except Exception as e:
        print(e)
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


def weighted_average(score: dict, confidence: dict) -> dict:
    """Compute weighted averages."""
    avg_score = {}

    metrics = [
        ("self_control_score", "self_control_confidence"),
        ("preparedness_score", "preparedness_confidence"),
        ("information_seeking_score", "information_seeking_confidence"),
        ("risk_seeking_score", "risk_seeking_confidence"),
        ("awareness_score", "awareness_confidence"),
        ("reaction_to_external_events_score", "reaction_to_external_events_confidence"),
    ]

    for score_key, conf_key in metrics:
        s_list = score.get(score_key, [])
        c_list = confidence.get(conf_key, [])

        # print("Processing:", score_key, conf_key)
        # print("s_list:", s_list)
        # print("c_list:", c_list)

        # Only compute if both score and confidence are valid lists with matching length
        if s_list and c_list and len(s_list) == len(c_list):
            denom = sum(c_list)
            if denom > 0:
                weighted_sum = sum(s * c for s, c in zip(s_list, c_list))
                # print("weighted_sum:", weighted_sum)
                # print("score_key:", score_key)

                avg_score[f"avg_{score_key}"] = weighted_sum / denom
                # print("avg_score[f'avg_{score_key}']:", avg_score[f"avg_{score_key}"])
            else:
                avg_score[f"avg_{score_key}"] = "Not assessed (zero confidence)"
        else:
            avg_score[f"avg_{score_key}"] = "Not assessed"

    # print("avg_score:", avg_score)
    return avg_score



def generate_persona_label(avg_score):

    print("Generate Persona Label Function Called")

    Awareness=""
    Self_Control=""
    Risk_Propensity=""

    if avg_score["avg_awareness_score"]=="Not assessed" or avg_score["avg_self_control_score"]=="Not assessed" or avg_score["avg_risk_seeking_score"]=="Not assessed":
        return "Not Assessed"

    for key, value in avg_score.items():
        
        if key=="avg_awareness_score":
            if value<2:
                Awareness="Basic awareness"
            elif value<3 and value>=2:
                Awareness="Developing Comprehension"
            elif value <4 and value>=3:
                Awareness= "Reasonable Understanding"
            else:
                Awareness="High"

        elif key=="avg_self_control_score":
            if value<2:
                Self_Control="Limited Restraint"
            elif value<3 and value>=2:
                Self_Control="Moderate Restraint"
            elif value<4 and value>=3:
                Self_Control="Reasonable Restraint"
            else:
                Self_Control="High"

        elif key=="avg_risk_seeking_score":
            if value<2.4:
                Risk_Propensity="Cautious"
            elif value<3.4 and value>=2.4:
                Risk_Propensity="Calculative"
            else:
                Risk_Propensity="Chance-Taking"


    key=(Awareness, Self_Control, Risk_Propensity)

    atire_mapping_dict={
        ("Basic awareness", "High", "Calculative"): "Calculative-Realist", 
        ("Basic awareness", "High", "Cautious"): "Cautious-Realist",
        ("Basic awareness", "High", "Chance-Taking"): "Chance-Taking-Realist",

        ("Basic awareness", "Limited Restraint", "Calculative"): "Calculative-Intuitionist",
        ("Basic awareness", "Limited Restraint", "Cautious"): "Cautious-Intuitionist",
        ("Basic awareness", "Limited Restraint", "Chance-Taking"): "Chance-Taking-Intuitionist",

        ("Basic awareness", "Moderate Restraint", "Calculative"): "Calculative-Aspirer",
        ("Basic awareness", "Moderate Restraint", "Cautious"): "Cautious-Aspirer",
        ("Basic awareness", "Moderate Restraint", "Chance-Taking"): "Chance-Taking-Aspirer",
        
        ("Basic awareness", "Reasonable Restraint", "Calculative"): "Calculative-Realist",
        ("Basic awareness", "Reasonable Restraint", "Cautious"): "Cautious-Realist",
        ("Basic awareness", "Reasonable Restraint", "Chance-Taking"): "Chance-Taking-Realist",
        
        ("Developing Comprehension", "High", "Calculative"): "Calculative-Info-Seeker",
        ("Developing Comprehension", "High", "Cautious"): "Cautious-Info-Seeker",
        ("Developing Comprehension", "High", "Chance-Taking"): "Chance-Taking-Info-Seeker",
        
        ("Developing Comprehension", "Limited Restraint", "Calculative"): "Calculative-Aspirer",
        ("Developing Comprehension", "Limited Restraint", "Cautious"): "Cautious-Aspirer",
        ("Developing Comprehension", "Limited Restraint", "Chance-Taking"): "Chance-Taking-Aspirer",

        ("Developing Comprehension", "Moderate Restraint", "Calculative"): "Calculative-Aspirer",
        ("Developing Comprehension", "Moderate Restraint", "Cautious"): "Cautious-Aspirer",
        ("Developing Comprehension", "Moderate Restraint", "Chance-Taking"): "Chance-Taking-Aspirer",
        
        ("Developing Comprehension", "Reasonable Restraint", "Calculative"): "Calculative-Info-Seeker",
        ("Developing Comprehension", "Reasonable Restraint", "Cautious"): "Cautious-Info-Seeker", 
        ("Developing Comprehension", "Reasonable Restraint", "Chance-Taking"): "Chance-Taking-Info-Seeker",
        
        ("Reasonable Understanding", "High", "Calculative"): "Calculative-Manager",
        ("Reasonable Understanding", "High", "Cautious"): "Cautious-Manager", 
        ("Reasonable Understanding", "High", "Chance-Taking"): "Chance-Taking-Manager",

        ("Reasonable Understanding", "Limited Restraint", "Calculative"): "Calculative-Explorer",
        ("Reasonable Understanding", "Limited Restraint", "Cautious"): "Cautious-Explorer",
        ("Reasonable Understanding", "Limited Restraint", "Chance-Taking"): "Chance-Taking-Explorer",

        ("Reasonable Understanding", "Moderate Restraint", "Calculative"): "Calculative-Discipline-seeker",
        ("Reasonable Understanding", "Moderate Restraint", "Cautious"): "Cautious-Discipline-seeker",
        ("Reasonable Understanding", "Moderate Restraint", "Chance-Taking"): "Chance-Taking-Discipline-seeker",

        ("Reasonable Understanding", "Reasonable Restraint", " Calculative"): "Calculative-Manager", 
        ("Reasonable Understanding", "Reasonable Restraint", "Cautious"): "Cautious-Manager",
        ("Reasonable Understanding", "Reasonable Restraint", "Chance-Taking"): "Chance-Taking-Manager",

        ("High", "High", "Calculative"): "Calculative-Strategist",
        ("High", "High", "Cautious"): "Cautious-Strategist",
        ("High", "High", "Chance-Taking"): "Chance-Taking-Strategist",

        ("High", "Limited Restraint", "Calculative"): "Calculative-Explorer",
        ("High", "Limited Restraint", "Cautious"): "Cautious-Explorer",
        ("High", "Limited Restraint", "Chance-Taking"): "Chance-Taking-Explorer",

        ("High", "Moderate Restraint", "Calculative"): "Calculative-Discipline-seeker",
        ("High", "Moderate Restraint", "Cautious"): "Cautious-Discipline-seeker",
        ("High", "Moderate Restraint", "Chance-Taking"): "Chance-Taking-Discipline-seeker",

        ("High", "Reasonable Restraint", "Calculative"): "Calculative-Manager", 
        ("High", "Reasonable Restraint", "Cautious"): "Cautious-Manager",
        ("High", "Reasonable Restraint", "Chance-Taking"): "Chance-Taking-Manager"
    }    
    persona_label=atire_mapping_dict.get(key, None)

    return persona_label

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
        print("Chat Request Received:")

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

        print("Current State before assessment graph")
        print(current_state)
        
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

        list_of_traits=["self_control","preparedness","information_seeking","risk_seeking","awareness","reaction_to_external_events"]

        score={}
        confidence={}
        rationale={}

        for trait in list_of_traits:
            score_key=f"{trait}_score"
            confidence_key=f"{trait}_confidence"
            rationale_key=f"{trait}_rationale"

            score[score_key]=serialized_state[score_key]
            confidence[confidence_key]=serialized_state[confidence_key]
            rationale[rationale_key]=serialized_state[rationale_key]

        ## Adding Persona label in state
        avg_score=weighted_average(score,confidence)
        persona_label=generate_persona_label(avg_score)
        
        print("avg_score:", avg_score)
        print("Persona Label:", persona_label)
        serialized_state["persona_label"]=persona_label

        #  Function for saving data in database

        
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

        if current_priority =='persona':
            persona_summary = serialized_state['persona']

            save_persona(
                user_id=user_id,
                session_id=session_id,
                persona_label=persona_label,
                persona=persona_summary,
                avg_score=avg_score,
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
        print(e)
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
        print(e)
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

        list_of_traits=["self_control","preparedness","information_seeking","risk_seeking","awareness","reaction_to_external_events"]

        score={}
        confidence={}
        rationale={}

        for trait in list_of_traits:
            score_key=f"{trait}_score"
            confidence_key=f"{trait}_confidence"
            rationale_key=f"{trait}_rationale"

            score[score_key]=current_state[score_key]
            confidence[confidence_key]=current_state[confidence_key]
            rationale[rationale_key]=current_state[rationale_key]

        avg_score=weighted_average(score,confidence)
        persona_label=generate_persona_label(avg_score)
        save_persona(
            user_id=user_id,
            session_id=session_id,
            persona_label=persona_label,
            persona=response_content,
            llm_confidence=confidence,
            parameter_score=score,
            avg_score=avg_score,
            parameter_rationale=rationale
        )

        return {
            "response": response_content,
            "persona_label": persona_label,
            "status": "success"
        }

    except Exception as e:
        print(e)
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
