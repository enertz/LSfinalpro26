import streamlit as st
from rag import init, answer

st.set_page_config(page_title="IITB Academic Assistant", page_icon="🎓")
st.title("🎓 IITB Academic Assistant")
st.caption("Ask me anything about IIT Bombay academics — courses, grading, exams, registration.")

@st.cache_resource
def load_rag():
    return init()

embed_model, gemini_model, index, chunks = load_rag()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about IITB academics...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = answer(question, index, chunks, embed_model, gemini_model)
        st.write(response)
        if sources:
            with st.expander("📄 Sources used"):
                for i, s in enumerate(sources):
                    st.markdown(f"**[{i+1}] {s['source']}** (score: {s['score']:.2f})")
                    st.caption(s['text'][:300] + "...")
    st.session_state.history.append({"role": "assistant", "content": response})