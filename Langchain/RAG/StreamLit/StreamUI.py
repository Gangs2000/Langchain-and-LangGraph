from DocRetrieval import execute_query
import streamlit as st
from typing import List, Any, Dict

def format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr)(doc, "metadata", None) or {}) is not None
    ]

st.set_page_config(page_title="Langchain documentation helper", layout="centered")
st.title("Langchain Documentation Helper")

# Sidebar logic
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    # Initialize the chat with a default assistant message
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": []
        }
    ]    

# Iterate through the messages and display them in the chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f" -{source}")

# Input area for user queries                 
prompt = st.chat_input("Ask a question about LangChain…")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "sources": []
        }
    )
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Retrieving docs and generating answer
    with st.chat_message("ai"):
        try:
            with st.spinner("Retrieving docs and generating answer"):
                result: Dict[str, Any] = execute_query(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer generated.)"
                sources = format_sources(result.get("context", []))
                
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")
            st.session_state.messages.append({
                "role":"assistant", 
                "content": answer,
                "sources": sources
            })
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)