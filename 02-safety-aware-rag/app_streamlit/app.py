import os, time, uuid, glob, re
import streamlit as st
from rag import SimpleRAG
from guardrails import violates
from telemetry import log
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../common")))
from model_clients import llm_complete


st.set_page_config(page_title="Safety-aware RAG", layout="wide")
st.title("Safety-aware RAG")

sid = st.session_state.get("sid") or str(uuid.uuid4())
st.session_state["sid"] = sid

threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)
k = st.slider("Top-K passages", 1, 10, 3)

# Load docs
docs = []
for p in glob.glob(os.path.join(os.path.dirname(__file__), "docs", "*.txt")):
    docs.append(open(p, encoding="utf-8").read())
if not docs:
    docs = ["This is a placeholder document about sample plan benefits with a deductible of $500."]

rag = SimpleRAG(docs)

q = st.text_input("Your question", "What is the deductible?")
if st.button("Ask"):
    t0 = time.time()
    hits = rag.retrieve(q, k=k)
    conf = hits[0][1] if hits else 0.0
    if conf < threshold:
        out = "I'm not fully confident. Could you specify the plan or add details?"
    else:
        context = "\n".join(h[-1] for h in hits)
        prompt = f"""Answer the user's question using ONLY the provided context. If insufficient, say you are unsure and ask a clarifying question.

Question: {q}

Context:\n{context}\n"""
        cand = llm_complete(prompt)
        out = f"{cand}\n\nSources: {', '.join([f'Doc#{i}' for i,_,_ in hits])}"
    if violates(out):
        out = "Refusing to answer due to policy (redacted sensitive content)."
    dt = int((time.time()-t0)*1000)
    st.write(out)
    log(sid, "RAG", "qna", dt, len(q.split()), len(out.split()))
