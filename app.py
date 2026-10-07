"""Bare Streamlit front end:  streamlit run app.py"""
import streamlit as st

from src.pipeline import RAG
from src.retrievers import ALL

COLOURS = {"SUPPORTED": "green", "REATTRIBUTED": "orange", "UNVERIFIED": "violet", "UNSUPPORTED": "red"}

st.title("citeguard")
retriever = st.selectbox("retriever", ALL, index=ALL.index("hybrid"))
nli = st.checkbox("verify with NLI cross-encoder instead of lexical")
two_stage = st.checkbox("two-stage: lexical check, then NLI on surviving claims")
question = st.text_input("question", "Are Roth IRA withdrawals taxed?")


@st.cache_resource
def rag(retriever: str, nli: bool, two_stage: bool) -> RAG:
    return RAG(full=True, retriever=retriever, nli=nli, two_stage=two_stage)


if st.button("ask") and question:
    ans = rag(retriever, nli, two_stage).ask(question, explain=True)
    st.metric("trust score", f"{ans.report.trust:.2f}")
    for v in ans.report.sentences:
        cite = f"[{v.cited}]" if v.status != "REATTRIBUTED" else f"[{v.cited}] → [{v.final}]"
        st.markdown(f":{COLOURS[v.status]}[**{v.status}** {v.score:.2f}] {v.sentence} {cite}")
    st.subheader("retrieved chunks")
    for i, (h, c) in enumerate(zip(ans.hits, ans.chunks), 1):
        with st.expander(f"[{i}] doc {h.doc_id} — score {h.score:.4f}"):
            st.write(c)
            st.json({k: round(s, 4) for k, s in h.terms.items()})
