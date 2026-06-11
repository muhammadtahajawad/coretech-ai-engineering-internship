"""
Task 5: Streamlit Chatbot User Interface
Intern Name: Muhammad Taha Jawad
Target Organization: CoreTech Innovations
"""

import streamlit as st
import pandas as pd
import re

# Set page config with no broken custom CSS backgrounds
st.set_page_config(
    page_title="CoreTech Innovations Portal",
    page_icon="🏢",
    layout="centered"
)

# Secure Data Loading
@st.cache_data
def load_and_normalize_dataset():
    try:
        raw_df = pd.read_csv("coretech_faq.csv")
        raw_df.columns = raw_df.columns.str.lower().str.strip()
        return raw_df.to_dict(orient="records")
    except Exception as data_error:
        return [
            {
                "question": "what services does coretech innovations offer?",
                "answer": "CoreTech Innovations delivers enterprise web development, custom mobile applications, scalable ERP systems, high-fidelity UI/UX design, zero-trust cybersecurity solutions, and targeted digital conversion marketing.",
                "keywords": "services offer build"
            },
            {
                "question": "what is coretech's contact information?",
                "answer": "You can connect with CoreTech Innovations directly via email at support@coretechio.com or visit coretechio.com.",
                "keywords": "contact email phone call support"
            }
        ]

faq_knowledge_base = load_and_normalize_dataset()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🏢 CoreTech Innovations")
    st.markdown("**System Status:** `OPERATIONAL`")
    st.markdown(f"**Knowledge Base:** `{len(faq_knowledge_base)} Profiles Loaded`")
    st.markdown("---")
    st.markdown("👤 **Muhammad Taha Jawad**")
    st.markdown("🎓 AI Engineering Track")

# Hybrid Matching Engine
def execute_hybrid_routing_engine(user_raw_query):
    clean_query = user_raw_query.lower().strip()
    query_tokens = set(re.findall(r'\b\w+\b', clean_query))
    
    if query_tokens.intersection({"contact", "email", "phone", "call", "support"}):
        return "📧 **CoreTech Contact Info:** Email us at `support@coretechio.com` or visit [coretechio.com](https://coretechio.com)."

    best_match_score = 0
    selected_system_response = "I want to make sure you get the exact answer you need. Please reach out to our team at support@coretechio.com or visit coretechio.com for immediate enterprise alignment."

    for profile in faq_knowledge_base:
        score = 0
        keywords_str = str(profile.get("keywords", "")).lower()
        question_str = str(profile.get("question", "")).lower()
        
        if clean_query in question_str or clean_query in keywords_str:
            score += 5
            
        kw_tokens = set(re.findall(r'\b\w+\b', keywords_str))
        score += len(query_tokens.intersection(kw_tokens)) * 2

        if score > best_match_score:
            best_match_score = score
            selected_system_response = profile.get("answer")

    return selected_system_response

# Main App UI Layout
st.title("🤖 CoreTech Innovations")
st.subheader("Enterprise Support Engine")
st.markdown("---")

if "chat_history_logs" not in st.session_state:
    st.session_state.chat_history_logs = [
        {"role": "assistant", "content": "Welcome to CoreTech Innovations. Ask me about our custom web platforms, mobile deployments, secure ERP systems, or corporate contact channels."}
    ]

# Render Chat
for packet in st.session_state.chat_history_logs:
    with st.chat_message(packet["role"]):
        st.write(packet["content"])

# User Input Loop
if client_prompt := st.chat_input("Enter your inquiry..."):
    with st.chat_message("user"):
        st.write(client_prompt)
    st.session_state.chat_history_logs.append({"role": "user", "content": client_prompt})
    
    output = execute_hybrid_routing_engine(client_prompt)
    
    with st.chat_message("assistant"):
        st.write(output)
    st.session_state.chat_history_logs.append({"role": "assistant", "content": output})