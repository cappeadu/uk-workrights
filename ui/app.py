"""Streamlit interface for the employment guidance assistant."""

from typing import Any

import streamlit as st


def display_response(response: dict[str, Any]) -> None:
    """Show an API response, including its status and source citations."""
    status = response.get("status")
    if status == "answer":
        st.success("Answer")
    elif status == "insufficient_evidence":
        st.warning("Not enough evidence")
    elif status == "out_of_scope":
        st.info("Outside the current topic scope")

    st.write(response.get("answer", ""))

    citations = response.get("citations", [])
    if not citations:
        st.caption("No source citations were returned.")
        return

    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            st.markdown(f"**{citation['title']}**")
            st.write(citation["excerpt"])
            st.link_button("Open GOV.UK source", citation["url"])


st.set_page_config(page_title="GOV.UK Employment Guidance", page_icon="📘")
st.title("GOV.UK Employment Guidance")
st.caption("General information from GOV.UK. This service is not legal advice.")

question = st.chat_input("Ask an employment question")
if question:
    st.session_state.pop("response", None)
    with st.chat_message("user"):
        st.write(question)

response = st.session_state.get("response")
if response:
    with st.chat_message("assistant"):
        display_response(response)
elif question:
    with st.chat_message("assistant"):
        st.info("The answer service will be connected in the next stage.")
else:
    st.write("Ask a question to get started.")
