# Financial assessment system using LangGraph for trait evaluation
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
import pprint
from pydantic import BaseModel, Field
from typing import Literal
import operator
import dotenv
import os
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()

def create_fallback_model():
    """Create a model with multiple API key fallbacks for reliability."""

    # Collect multiple Gemini API keys for fallback
    api_keys = []
    for i in range(1, 21):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        # print("api_key")
        # print(key)
        if key:
            api_keys.append(key)

    main_key = os.getenv("GEMINI_API_KEY")
    if main_key:
        api_keys.append(main_key)

    if not api_keys:
        raise ValueError("No GEMINI API keys found in environment variables")

    # Use OpenAI as primary model
    primary_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            max_retries=2,
            api_key=main_key
        )

    # Create Gemini fallback models
    fallback_models = []
    for key in api_keys:
        fallback_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            max_retries=2,
            api_key=key
        )
        fallback_models.append(fallback_model)

    if fallback_models:
        return primary_model.with_fallbacks(
            fallbacks=fallback_models,
            exceptions_to_handle=(Exception,)
        )
    else:
        return primary_model


generator_model = create_fallback_model()

# Financial assessment questions organized by trait
QUESTIONS = {
    "awareness": [
        "Do you know how much you need to save regularly to meet your financial goals?",
        "Do you know how to choose the right insurance policy for your needs?",
        "Are you aware that an investment with high return is usually associated with high risk?",
        "Do you know how to save effectively to ensure a peaceful retired life?",
        "Are you aware that buying a single company's stock can be riskier than investing in a mutual fund?",
        "If the interest rate on your savings account is 1% per year and inflation is 2% per year, would you be able to buy more than, exactly the same, or less than today after one year with the same amount of money?"
    ],
    "self_control": [
        "Do you tend to spend money when you're unhappy or stressed?",
        "Do special offers often tempt you into buying things you don’t really need?",
        "Do you pay your bills on time?",
        "When making financial decisions, do you reflect carefully or act impulsively?"
    ],
    "preparedness": [
        "How confident are you that you could handle a major financial emergency?",
        "If you lost your main source of income, for how long could you cover your expenses without borrowing money?",
        "How much stress do you feel about the future of your financial situation?"
    ],
    "information_seeking": [
        "Which sources do you rely on when making financial decisions (e.g., internet, financial advisors, friends/family)?",
        "How frequently do you seek out information before making a major financial decision or large purchase?"
    ],
    "risk_seeking": [
        "What level of financial risk have you taken with your investments so far?",
        "What level of financial risk are you currently comfortable with?",
        "Compared to others, how would you rate your willingness to take financial risks?",
        "Have you ever invested a large amount in a risky asset mainly for the thrill of watching how it performs?",
        "When you hear the term 'financial risk', what comes to your mind?"
    ],
    "reaction_to_external_events": [
        "How quickly do you recover from a stressful financial event?"
    ]
}

# Detailed definitions for each financial trait being assessed
DEFINITIONS = {
    "awareness": (
        "Financial awareness refers to an individual's understanding of key financial concepts such as saving strategies, risk-return tradeoffs, inflation effects, and investment diversification. "
        "It reflects the extent to which a person is financially literate and can make informed decisions by anticipating outcomes based on available information. "
        "Assessment involves evaluating knowledge and awareness of core financial principles and the ability to apply them in practical scenarios."
        "Elicit response about awareness by asking questions whether and how much they understand about these principles. Understanding on how to invest and where to invest is a good indicator of financial awareness. Try to gauge their understanding on both objective and subjective awareness."
    ),
    "self_control": (
        "Self-control in financial settings denotes the ability to regulate impulsive spending, delay gratification, and make deliberate financial choices despite emotional triggers or external temptations. "
        "It reflects behavioral regulation under affective or marketing-driven stimuli. "
        "Assessment focuses on patterns of impulsive purchases, emotional spending behavior, and adherence to planned financial commitments."
    ),
    "preparedness": (
        "Financial preparedness captures the degree to which an individual is equipped to handle unexpected financial shocks, such as income loss or emergencies, through planning and resource allocation. "
        "It encompasses both objective readiness (e.g., emergency funds) and subjective confidence in coping with financial adversity. "
        "Assessment centers on resource buffers, temporal self-sufficiency, and perceived resilience toward future uncertainties."
        "Elicit response about preparedness by asking questions about their confidence in handling financial emergencies, their savings for such events. Insurance coverage can also be a good indicator of preparedness."
    ),
    "information_seeking": (
        "Information seeking refers to the proactive behavior of acquiring relevant data, advice, or insights prior to making financial decisions. "
        "It reflects epistemic curiosity, openness to expertise, and diligence in reducing uncertainty before committing resources. "
        "Assessment involves measuring reliance on diverse information channels and the frequency or depth of pre-decision inquiry."
    ),
    "risk_seeking": (
        "Risk seeking represents an individual's tendency to prefer or tolerate uncertainty in pursuit of potentially higher financial returns. "
        "It involves both cognitive risk tolerance and affective attraction to volatility or thrill. "
        "Assessment includes historical investment behavior, self-reported risk appetite, and attitudes toward uncertain outcomes in monetary contexts."
    ),
    "reaction_to_external_events": (
        "Reaction to external events measures emotional resilience and adaptability in response to financial stressors or market volatility. "
        "It reflects how quickly and effectively an individual recovers from financial setbacks or unanticipated disruptions. "
        "Assessment is based on emotional stability indicators, stress recovery duration, and behavioral adaptation post adverse financial events."
    )
}

# Assessment configuration
TRAITS = ["awareness", "self_control", "preparedness",
          "information_seeking", "risk_seeking", "reaction_to_external_events"]
PRIORITY_TRAITS = {value: index+1 for index,
                   value in enumerate(TRAITS)}  # Trait priority mapping
# Maximum questions per trait [keep in mind the starting index is 0 for most traits a-2 for awareness, modify in LOCAL_STATE and EMPTY_STATE]
MAX_QUESTIONS = 3
CONFIDENCE_THRESHOLD = 7  # Minimum confidence score to move to next trait


class AgentState(TypedDict):
    """State management for the financial assessment agent"""
    current_priority: Literal["awareness", "self_control", "preparedness",
                              "information_seeking", "risk_seeking", "reaction_to_external_events"]
    current_iteration: int
    assessment_type: Literal["conversation_based", "option_based"]

    # Score tracking for each trait
    self_control_score: Annotated[list, operator.add]
    preparedness_score: Annotated[list, operator.add]
    information_seeking_score: Annotated[list, operator.add]
    risk_seeking_score: Annotated[list, operator.add]
    awareness_score: Annotated[list, operator.add]
    reaction_to_external_events_score: Annotated[list, operator.add]

    # Confidence tracking for each trait assessment
    self_control_confidence: Annotated[list, operator.add]
    preparedness_confidence: Annotated[list, operator.add]
    information_seeking_confidence: Annotated[list, operator.add]
    risk_seeking_confidence: Annotated[list, operator.add]
    awareness_confidence: Annotated[list, operator.add]
    reaction_to_external_events_confidence: Annotated[list, operator.add]

    # Extracted sentences from user responses
    self_control_sentences: Annotated[list, operator.add]
    preparedness_sentences: Annotated[list, operator.add]
    information_seeking_sentences: Annotated[list, operator.add]
    risk_seeking_sentences: Annotated[list, operator.add]
    awareness_sentences: Annotated[list, operator.add]
    reaction_to_external_events_sentences: Annotated[list, operator.add]

    # Rationale for each assessment
    self_control_rationale: Annotated[list, operator.add]
    preparedness_rationale: Annotated[list, operator.add]
    information_seeking_rationale: Annotated[list, operator.add]
    risk_seeking_rationale: Annotated[list, operator.add]
    awareness_rationale: Annotated[list, operator.add]
    reaction_to_external_events_rationale: Annotated[list, operator.add]

    messages: Annotated[list, add_messages]
    redirect_path: Literal["evaluator", "refocus"]
    trait_evaluation: Literal["awareness", "self_control", "preparedness",
                              "information_seeking", "risk_seeking", "reaction_to_external_events"]
    persona: str
    persona_label: str
    continue_conversation: bool


# Initial state templates
LOCAL_STATE: AgentState = {
    "current_priority": "awareness",
    "current_iteration": -2,
    "assessment_type": "conversation_based",

    "awareness_score": [],
    "self_control_score": [],
    "preparedness_score": [],
    "information_seeking_score": [],
    "risk_seeking_score": [],
    "reaction_to_external_events_score": [],

    "awareness_confidence": [],
    "self_control_confidence": [],
    "preparedness_confidence": [],
    "information_seeking_confidence": [],
    "risk_seeking_confidence": [],
    "reaction_to_external_events_confidence": [],

    "awareness_sentences": [],
    "self_control_sentences": [],
    "preparedness_sentences": [],
    "information_seeking_sentences": [],
    "risk_seeking_sentences": [],
    "reaction_to_external_events_sentences": [],

    "awareness_rationale": [],
    "self_control_rationale": [],
    "preparedness_rationale": [],
    "information_seeking_rationale": [],
    "risk_seeking_rationale": [],
    "reaction_to_external_events_rationale": [],

    "messages": [],
    "redirect_path": "evaluator",
    "trait_evaluation": "awareness",
    "persona": "",
    "persona_label": "Not Assessed",
    "continue_conversation": True
    # "messages": []
}

EMPTY_STATE: AgentState = {
    "current_priority": "awareness",
    "current_iteration": -2,
    "assessment_type": "conversation_based",

    "awareness_score": [],
    "self_control_score": [],
    "preparedness_score": [],
    "information_seeking_score": [],
    "risk_seeking_score": [],
    "reaction_to_external_events_score": [],

    "awareness_confidence": [],
    "self_control_confidence": [],
    "preparedness_confidence": [],
    "information_seeking_confidence": [],
    "risk_seeking_confidence": [],
    "reaction_to_external_events_confidence": [],

    "awareness_sentences": [],
    "self_control_sentences": [],
    "preparedness_sentences": [],
    "information_seeking_sentences": [],
    "risk_seeking_sentences": [],
    "reaction_to_external_events_sentences": [],

    "awareness_rationale": [],
    "self_control_rationale": [],
    "preparedness_rationale": [],
    "information_seeking_rationale": [],
    "risk_seeking_rationale": [],
    "reaction_to_external_events_rationale": [],

    "messages": [],
    "redirect_path": "evaluator",
    "trait_evaluation": "awareness",
    "persona": "",
    "persona_label": "Not Assessed",
    "continue_conversation": True
    # "messages": []
}


def format_messages_for_prompt(messages):
    """Convert message list to formatted string for prompts"""
    formatted_messages = ""
    if not messages or len(messages) == 0:
        return "No previous messages, start the conversation:"
    for message in messages:
        if message.type == "human":
            formatted_messages += f"User: {message.content}\n"
        elif message.type == "ai":
            formatted_messages += f"{message.content}\n"
    return formatted_messages


def prioritize_trait(state: AgentState):
    """Determine next trait to assess based on confidence and iteration count"""
    print("prioritizing trait to assess\n")
    current_priority = state["current_priority"]
    current_iteration = state["current_iteration"]
    # print(f"current state:\n")
    # Calculate average confidence for current trait
    confidence = sum(state[f"{current_priority}_confidence"])/len(state[f"{current_priority}_confidence"]
                                                                  ) if state[f"{current_priority}_confidence"] and len(state[f"{current_priority}_confidence"]) > 0 else 0
    # print(f"current average confidence: {confidence}")
    # print(f"MAX_QUESTIONS: {MAX_QUESTIONS}")
    # print(f"current iteration: {current_iteration}")
    # print(f"current priority: {current_priority}")
    # print(f"TRAITS: {TRAITS}")
    # Check if assessment is complete
    if current_iteration >= MAX_QUESTIONS and current_priority == TRAITS[-1]:
        print("Assessment complete, generating persona.")
        return {"current_priority": "persona", "continue_conversation": False}
    # Move to next trait if confidence threshold met
    elif confidence >= CONFIDENCE_THRESHOLD and current_iteration > MAX_QUESTIONS -1:
        print("Moving to next trait based on confidence.")
        return {"current_priority": TRAITS[PRIORITY_TRAITS[current_priority]], "assessment_type": "conversation_based", "current_iteration": 0}
    # Switch to option-based assessment if confidence low
    elif current_iteration == MAX_QUESTIONS - 1 and confidence < CONFIDENCE_THRESHOLD:
        print("Switching to option-based assessment for current trait.")
        return {"assessment_type": "option_based", "current_iteration": MAX_QUESTIONS}
    # Move to next trait after max questions
    elif current_iteration == MAX_QUESTIONS and current_priority != TRAITS[-1]:
        print("Max questions reached, moving to next trait.")
        return {"current_priority": TRAITS[PRIORITY_TRAITS[current_priority]], "assessment_type": "conversation_based", "current_iteration": 0}
    else:
        print("Continuing with current trait.")
        return {"current_iteration": current_iteration + 1}


modified_assessment_question_prompt = """
You are a financial expert with excellent communication skills, and can assess any person about their financial traits based on a natural conversation.
Your are friendly, charming and confident and can get anyone to naturally open up.

**Hidden Steering Data** (do *not* mention in output):

Hidden Context (model, do not surface):
trait_to_assess:  {trait_to_assess}
assessment_type:  {assessment_type}
trait_definition: {trait_definition}
trait_guidance:   {trait_questions}

Your task is to continue this conversation in such a friendly and natural way that the user doesn't feel like they are getting assessed and it just feels like a friendly conversation.
If the user has went fully off topic in some way, or not replying with any helpful information, gently steer the convesation in a way that you can assess the trait form their response.

Always maintain the natural flow of conversation, even when switching to different trait assessment or different question, and never break character.
Be kind and helpful.

Never ask the user to self assess based on any scale. Your only task is to continue the flow of conversation, guiding the flow towards the required trait to assess.

Always ask one question at a time.

{assessment_information}

ONGOING MESSAGE THREAD:
{messages}
"""


def assessment_conversation_generator(state: AgentState):
    """Generate conversation or option-based questions for trait assessment"""
    print(
        f"current_priority:{state['current_priority']}\nassessment_type:{state['assessment_type']}\ncurrent_iteration:{state['current_iteration']}")
    trait_to_assess = state["current_priority"]
    assessment_type = state["assessment_type"]
    print("generating assessment question\n")
    print("should continue: ", state["continue_conversation"])

    # Generate persona if assessment is complete
    if state["current_priority"] == "persona" and not state["continue_conversation"]:
        persona = generate_persona(state)
        return {"persona": persona}

    # Determine assessment approach based on type
    assessment_information = f"""
    The assessment type is option_based, ask option based question, continuing the flow of conversation, like prompting user to choose what of the option resonates with them the most, out of a few options.
    The option based question should be specifically targeted to extract the financial trait {trait_to_assess} of the user, based on the option they choose.
    The options should be clear, and concise so that a conclusion about the user's trait can be made.
    The option should be given in format like
    A) ...\n
    B) ...\n
    The option question must be based on the exact of questions for assessing the {trait_to_assess}.
    """ if assessment_type == "option_based" else "Continue the natural conversation based assessment if the assessment type is conversation_based."

    formatted_assessment_question_prompt = modified_assessment_question_prompt.format(trait_to_assess=state["current_priority"], assessment_type=state["assessment_type"], messages=format_messages_for_prompt(
        state["messages"]), trait_definition=DEFINITIONS[state["current_priority"]], trait_questions=QUESTIONS[state["current_priority"]], assessment_information=assessment_information)

    # print("format_messages_for_prompt")
    # print(format_messages_for_prompt(state["messages"]))
    # save_message(user_id,session_id,"system",format_messages_for_prompt(state["messages"]))
    # print(messages.find_one({"session_id":1}))
    result = generator_model.invoke(formatted_assessment_question_prompt)
    # print("response")
    # print(result)
    return {"messages": [result]}

# assessment_conversation_generator(LOCAL_STATE)


class RationaleResponse(BaseModel):
    """Response model for extracting trait rationale from user responses"""
    trait: Literal["awareness", "self_control", "preparedness",
                   "information_seeking", "risk_seeking", "reaction_to_external_events"]
    rationale: str
    sentence: str


def sentence_and_rationale_extractor(state: AgentState):
    """Extract key sentences and rationale from user responses for trait assessment"""
    print("generating rationale\n")
    rationale_prompt = """
    You are an expert psychologist with experience in understanding financial traits of a person from their natural conversation.
    Your task is, given a conversation, to analyze last USER TURN and output **exactly** three fields:

    TRAIT CONTEXT:
    {definitions}

    TRAIT_TO_ASSESS:
    {trait_to_be_assessed}

    PRIMARY ASSESSMENT STRATEGY:
    1. **First Priority**: Assess the trait "{trait_to_be_assessed}" based on the user's last message
       - Examine if the user's response provides any information about their {trait_to_be_assessed}
       - Look for direct or indirect indicators of this specific trait in their reply
       - If you can reliably extract information about {trait_to_be_assessed} from the last user message, use this trait

    2. **Fallback Strategy**: If {trait_to_be_assessed} cannot be assessed from the user's response:
       - Identify which of the 6 traits ["awareness", "self_control", "preparedness", "information_seeking", "risk_seeking", "reaction_to_external_events"] 
         is most clearly expressed in the user's last message
       - Use the trait that has the strongest evidence in the user's response

    OUTPUT FIELDS:

    1. **trait**
       - If {trait_to_be_assessed} can be assessed from the user's response: return "{trait_to_be_assessed}"
       - If {trait_to_be_assessed} cannot be assessed: return the most probable trait from the 6 options
       - If no trait can be reliably extracted or no financial information is present: return empty string

    2. **sentence**
       – A concise paraphrase of what the user's reply *means* in terms of their intended action or stance.
       – Format: single sentence, present‑tense, starting with "User will …" or "User feels …".
       - Even if the user has just replied yes, or no, or chooses an option based on the previous question, gather the complete context from the previous message and create the complete sentence,
         which has the entire information of what user agreed to or denied or meant when he chose an option, or said yes or no.
       – **Max 20 words.**

    3. **rationale**
       – A hypothesized explanation of *why* the user answered that way, grounded **only** in the single trait being assessed.
       – Format: 1–2 sentences, referencing **exactly one** trait name in braces (e.g., `{trait_to_be_assessed}` or the fallback trait), and linking to underlying values or beliefs.
       – **Max 20–25 words.**

    HARD CONSTRAINTS (do not violate):
    - Process **only** the user's last turn; you may use the question for understanding the full context.
    - Output **only** valid JSON, with keys `"trait"`, `"sentence"` and `"rationale"`.
    - Do **not** surface raw definitions or questions.
    - Always reference **exactly one** trait tag in the rationale.
    - **If the last user turn contains no extractable content relevant to any trait**, or the user reply does not map to a valid action or stance, return empty strings for all fields.
    - If the last user response is not related to the financial context or irrelevant to the ongoing conversation, return empty strings for all fields.

    ONGOING MESSAGE THREAD:
    {messages}
    """

    structured_model = generator_model.with_structured_output(
        RationaleResponse)
    formatted_prompt = rationale_prompt.format(
        trait_to_assess=state["current_priority"], definitions=DEFINITIONS, messages=format_messages_for_prompt(state["messages"]), trait_to_be_assessed=state["current_priority"])

    response = structured_model.invoke(formatted_prompt)
    print(response)

    # Skip if no valid content extracted
    if not response.sentence or not response.rationale or response.sentence == "" or response.rationale == "":
        return

    trait = response.trait
    return {f"{trait}_sentences": [response.sentence], f"{trait}_rationale": [response.rationale], "trait_evaluation": trait}

# sentence_and_rationale_extractor(LOCAL_STATE)


class ScoreResponse(BaseModel):
    """Response model for trait scoring and confidence assessment"""
    score: int
    confidence: int


def score_and_confidence_extractor(state: AgentState):
    """Generate numerical scores and confidence levels for trait assessments"""
    print("generating score\n")
    scoring_prompt = """
You are a “Scoring Engine” whose job is to evaluate the User’s level on a single financial trait—using the full café mini‑scene, the extracted sentence, and its rationale—and to assign:

1. **score**
   – An integer 1–5 on a Likert scale, where:
     1 = very low trait expression
     2 = low
     3 = moderate
     4 = high
     5 = very high

2. **confidence**
   – An integer 1–10 indicating your confidence in the score (1 = very uncertain; 10 = very certain).

HARD CONSTRAINTS (do not violate):
- Process **only** the inputs provided below.
- Output **only** valid JSON with keys `"score"` and `"confidence"`.
- Both values must be integers in the specified ranges.
- If there is no valid user content to assess, return 0 for score and confidence.

ONGOING MESSAGE THREAD:
{messages}

TRAIT TO ASSESS:
{trait_evaluation}

TRAIT DEFINITION:
{trait_definition}

sentence:
{sentence}

RATIONALE:
{rationale}

IMPORTANT: ONLY FIND RATIONALE FROM THE LAST MESSAGE BY THE USER. USE ONLY PREVIOUS MESSAGE IF NECESSARY FOR CONTEXT BUT THE RATIONALE AND sentence SHOULD ONLY BE FROM THE LAST USER MESSAGE.
If the sentence and rationale are empty, return 0 for score and confidence.
"""

    structured_model = generator_model.with_structured_output(ScoreResponse)

    trait = state["trait_evaluation"]
    utter_list = state.get(f"{trait}_sentences", [])
    rationale_list = state.get(f"{trait}_rationale", [])

    print(f" assessing {trait} scores:")

    # Get the most recent sentence and rationale for scoring
    last_sentence = utter_list[-1] if utter_list else ""
    last_rationale = rationale_list[-1] if rationale_list else ""

    formatted_score_extractor_prompt = scoring_prompt.format(messages=format_messages_for_prompt(
        state["messages"]), trait_evaluation=state["trait_evaluation"], rationale=last_rationale, sentence=last_sentence, trait_definition=DEFINITIONS[state["current_priority"]])
    result = structured_model.invoke(formatted_score_extractor_prompt)
    print(result)

    # Skip if no valid assessment
    if result.confidence == 0 or result.score == 0:
        return

    return {f"{trait}_score": [result.score], f"{trait}_confidence": [result.confidence]}
# score_and_confidence_extractor(LOCAL_STATE)


class ValidationResponse(BaseModel):
    """Response model for validating user response relevance"""
    redirect_path: Literal["evaluator", "refocus"] = Field(
        description="return 'evaluator' if the user's response contains information that can be used to extract financial traits of the user, otherwise return 'refocus'")


def response_validator(state: AgentState):
    """Validate if user response contains relevant financial trait information"""
    print("validating response\n")

    system_prompt = """
    Traits to assess and their definitions:
    {definitions}

    Ongoing Message Thread:
    {messages}

    Trait currently trying to assess:
    {trait_to_assess}
    You have been provided with the current ongoing message thread between user and AI.
    Your task is to critically analyse the flow of the conversation and identify whether the last  response of the user is in the correct direction, or if the user is getting distracted,
    or if the user is the response of user irrelevant in assessing the financial traits.

    Determine critically by understanding what the user is trying to say using the last few conversation based on the message provided. Even if the user has only said yes, orno or maybe choose an option,
    from the conversation going on, verify what the user meant and if any information about financial trait can be extracted from that, then return "redirect_path":"evaluator" in python dictionary form with brackets.

    return "redirect_path":"evaluator if the conversation has important information regarding the user that can be helpful in extracting their financial traits in python dictionary form with brackets
    Otherwise return "redirect_path":"refocus" in python dictionary form with brackets

    If no relevant information is available, return "redirect_path":"refocus" in python dictionary form with brackets.
    """
    formatted_prompt = system_prompt.format(messages=format_messages_for_prompt(
        state["messages"]), trait_to_assess=state["current_priority"], definitions=DEFINITIONS)
    validator_model = generator_model.with_structured_output(
        ValidationResponse)

    result = validator_model.invoke(formatted_prompt)
    print(result)

    return {"redirect_path": result.redirect_path}


def router(state: AgentState):
    """Route to appropriate node based on response validation"""
    if state["redirect_path"] == "evaluator":
        print("redirecting to sentence and rationale extractor node")
        return "sentence_and_rationale_extractor"
    elif state["redirect_path"] == "refocus":
        print("redirecting to trait prioritization node")
        return "prioritize_trait"


# def persona_generator(state: AgentState):
#     print("generating persona\n")
#     persona = generate_persona(state)
#     print(persona)

#     return {"messages": [persona]}


# Build the assessment workflow graph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("sentence_and_rationale_extractor",
               sentence_and_rationale_extractor)
graph.add_node("score_and_confidence_extractor",
               score_and_confidence_extractor)
graph.add_node("prioritize_trait", prioritize_trait)
graph.add_node("assessment_conversation_generator",
               assessment_conversation_generator)
graph.add_node("response_validator", response_validator)

# Define graph flow
graph.set_entry_point("response_validator")
graph.add_conditional_edges("response_validator", router, {
                            "sentence_and_rationale_extractor": "sentence_and_rationale_extractor", "prioritize_trait": "prioritize_trait"})
graph.set_finish_point("assessment_conversation_generator")

# Connect nodes with edges
graph.add_edge("sentence_and_rationale_extractor",
               "score_and_confidence_extractor")
graph.add_edge("score_and_confidence_extractor", "prioritize_trait")
graph.add_edge("prioritize_trait", "assessment_conversation_generator")

assessment_graph = graph.compile()

# Reset state to empty template
LOCAL_STATE = {**EMPTY_STATE}


def run_chat():
    """Interactive chat function for testing the assessment system"""
    global LOCAL_STATE
    while True:
        user_prompt = input()
        if user_prompt == "quit":
            break
        elif user_prompt == "state":
            pprint.pp({key: value for key, value in LOCAL_STATE.items()
                      if value != 0 and key != "messages" and value != []})
            continue
        LOCAL_STATE["messages"].append(HumanMessage(content=user_prompt))
        NEW_STATE = assessment_graph.invoke(LOCAL_STATE)
        LOCAL_STATE = NEW_STATE

        for msg in LOCAL_STATE["messages"][-2:]:
            msg.pretty_print()


def generate_persona(state: AgentState):
    """Generate final financial persona summary based on assessment results"""
    print("generating persona\n")

    system_prompt = """
You are a financial psychology analyst AI.

Definitions:
{definitions}

You will be given:
- A list of previous messages in a conversation, including both the user's dialogue and any prior assistant responses.
- A rationale field that summarizes the underlying behavioral or financial reasoning inferred from the user's responses.
- A persona label to display to user.

Your task is to synthesize this information into a **concise, structured financial persona summary** that includes the following sections:

1. **Financial Persona Summary**:  
   - Provide a high-level description of the user’s financial thinking style, habits, tendencies, and mindset based on the dialogue so far.  
   - Avoid generic statements—anchor your conclusions in *evidence from the conversation*, including concrete behaviors or attitudes the user expressed.
   
2. **Strengths**:  
   - Clearly enumerate what financial strengths the user has (e.g., planning, long-term thinking, impulse control, flexibility).
   - Explain *why* these are strengths, backing it with references to the user's statements or behavior patterns.
   - Do not comment about the risk seeking ability, or reaction to external events as strength or weakness.

3. **Weaknesses**:  
   - Clearly describe one or more financial blind spots or behavioral weaknesses the user may have.
   - Do not sugarcoat. Use direct language but remain constructive.
   - Justify the weakness using signals from the conversation (e.g., choices, hesitation, contradictions, overconfidence, etc.)
   - Do not comment about the risk seeking ability, or reaction to external events as strength or weakness.

4. **Suggestions for Improvement**:  
   - Give **practical and actionable** recommendations tailored to the user’s weaknesses.
   - These should not be generic tips, but rooted in behavioral finance or psychology, and framed to the user's persona.
   - Include both short-term habits and long-term mindset shifts if applicable.

   IMPORTANT: - Do not comment about the risk seeking ability, or reaction to external events as strength or weakness.

**Output Style Guide**:
- Keep persona label at the top in the format: "Persona Label":<Label>. If the persona label is "Not Assessed", do not include it in the output.
- Knowledge on each trait should be demonstrated for Summary and Strengths/Weaknesses/SuggestionForImprovement sections.
- Be concise but information-rich. 
- Avoid vague praise or platitudes.
- Each section should be **clearly titled**, e.g., `Strengths`, `Weaknesses`, etc.
- Write in professional, psychologically grounded language, with clinical clarity and confidence.
- Write all the sections in second person for the user, not in third person, as user.

Ongoing Message Thread: 
{messages}

Persona Label to Display: 
{persona_label}

Extracted Sentences:
{extracted_sentences}

Extracted Rationales:
{extracted_rationales}
"""
    formatted_system_prompt = system_prompt.format(definitions=DEFINITIONS, messages=format_messages_for_prompt(state["messages"]), extracted_sentences=[
                                                   state[f"{trait}_sentences"] for trait in TRAITS], extracted_rationales=[state[f"{trait}_rationale"] for trait in TRAITS], persona_label=state["persona_label"])

    response = generator_model.invoke(formatted_system_prompt)
    return response.content
