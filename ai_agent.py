# ================================================================
#   AI MEMORY AGENT — FINAL PROJECT
#   Built with: LangChain + LangGraph + MongoDB + SerpAPI + Streamlit
#   Features:
#   - Chat with AI using Google Gemini
#   - Real-time internet search via SerpAPI
#   - Long-term memory via MongoDB
#   - Human in the Loop confirmation
#   - Chat history display
# ================================================================

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from serpapi import GoogleSearch
from langchain.tools import tool
import streamlit as st

# ─────────────────────────────────────────
# STEP 1 — Page Config & CSS Styling
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Memory Agent",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .stButton button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        width: 100%;
    }
    h1 { color: white !important; text-align: center; }
    .stTextInput input { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# STEP 2 — Load API Keys from Secrets
# ─────────────────────────────────────────
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDtu62RqGc6vag8RZmNFnH1zpnomXQR_oU")
SERP_API_KEY   = st.secrets.get("SERP_API_KEY",   "748f45d5d43b2ea0bc6edaf20ccdeb4807fea47e35e9689781d4ac0042df87a0")
MONGODB_URI    = st.secrets.get("MONGODB_URI",    "mongodb+srv://khnbutt118_db_user:QLaFKV8usDNC9VEd@cluster0.7s3p0bb.mongodb.net/")

# ─────────────────────────────────────────
# STEP 3 — Setup Gemini LLM
# ─────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0
)

# ─────────────────────────────────────────
# STEP 4 — Define Search Tool (SerpAPI)
# ─────────────────────────────────────────
@tool
def search_internet(query: str) -> str:
    """
    Search the internet for real-time information.
    Use this tool when the user asks about current events,
    weather, news, or anything requiring up-to-date data.
    """
    params = {
        "q"      : query,
        "hl"     : "en",
        "gl"     : "us",
        "api_key": SERP_API_KEY
    }
    search  = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" in results:
        top = results["organic_results"][0]
        return f"{top['title']}: {top.get('snippet', 'No details available')}"
    return "No results found for this query."

# ─────────────────────────────────────────
# STEP 5 — Setup MongoDB Memory + Agent
# ─────────────────────────────────────────
if "agent" not in st.session_state:
    mongo_client              = MongoClient(MONGODB_URI)
    checkpointer              = MongoDBSaver(mongo_client)
    st.session_state.agent    = create_react_agent(
        model      = llm,
        tools      = [search_internet],
        checkpointer = checkpointer
    )

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize human in the loop state
if "pending_response" not in st.session_state:
    st.session_state.pending_response = None

if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False

# ─────────────────────────────────────────
# STEP 6 — Streamlit UI
# ─────────────────────────────────────────
st.title("AI Memory Agent")
st.markdown(
    "<p style='text-align:center; color:#aaa;'>"
    "Chat with AI that remembers your conversations "
    "and searches the internet in real-time</p>",
    unsafe_allow_html=True
)

st.divider()

# Sidebar — session settings
with st.sidebar:
    st.header("⚙️ Settings")
    thread_id = st.text_input(
        "Your Name (Session ID):",
        value="user_123",
        help="This keeps your conversation history separate from others"
    )
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("- Type any question below")
    st.markdown("- Agent searches internet when needed")
    st.markdown("- Your conversation is saved in MongoDB")
    st.markdown("- Change your name to start a new session")
    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history      = []
        st.session_state.pending_response  = None
        st.session_state.awaiting_confirmation = False
        st.rerun()

# ─────────────────────────────────────────
# STEP 7 — Display Chat History
# ─────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ─────────────────────────────────────────
# STEP 8 — Human in the Loop Section
# If agent is unsure, ask user for confirmation
# ─────────────────────────────────────────
if st.session_state.awaiting_confirmation:
    st.warning("⚠️ The agent is unsure about this response. Please review:")
    st.info(st.session_state.pending_response)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm — Looks Good"):
            # User approved — show the response
            st.session_state.chat_history.append({
                "role"   : "assistant",
                "content": st.session_state.pending_response
            })
            st.session_state.awaiting_confirmation = False
            st.session_state.pending_response      = None
            st.rerun()
    with col2:
        if st.button("❌ Reject — Ask Again"):
            # User rejected — discard the response
            st.session_state.awaiting_confirmation = False
            st.session_state.pending_response      = None
            st.warning("Response rejected. Please ask your question again.")
            st.rerun()

# ─────────────────────────────────────────
# STEP 9 — Chat Input
# ─────────────────────────────────────────
else:
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        # Show user message immediately
        st.session_state.chat_history.append({
            "role"   : "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.write(user_input)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking and searching..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config={"configurable": {"thread_id": thread_id}}
                )

            # Extract clean text from response
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                content = last_message.content
                if isinstance(content, list):
                    text = " ".join([
                        c["text"] for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ])
                else:
                    text = content
            else:
                text = str(last_message)

            # ─────────────────────────────
            # HUMAN IN THE LOOP CHECK
            # If agent response contains uncertainty phrases
            # ask user for confirmation before showing it
            # ─────────────────────────────
            uncertainty_phrases = [
                "i'm not sure",
                "i am not sure",
                "i don't know",
                "i do not know",
                "i cannot confirm",
                "i'm uncertain",
                "not certain",
                "you may want to verify",
                "please confirm",
                "i'm unsure"
            ]

            is_uncertain = any(
                phrase in text.lower()
                for phrase in uncertainty_phrases
            )

            if is_uncertain:
                # Agent is unsure — trigger human in the loop
                st.session_state.pending_response      = text
                st.session_state.awaiting_confirmation = True
                st.rerun()
            else:
                # Agent is confident — show response directly
                st.write(text)
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": text
                })