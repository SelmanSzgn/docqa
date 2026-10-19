import os

import requests
import streamlit as st

API_URL = os.environ.get("DOCQA_API_URL", "http://127.0.0.1:8000/ask/sources")

st.title("docqa")
question = st.text_input("Question")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                API_URL, json={"question": question}, timeout=180
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as error:
            st.error(f"API error: {error}")
        else:
            data = response.json()
            st.write(data["answer"])
            st.subheader("Sources")
            for source in data["sources"]:
                st.write(f"- {source['source']}, p.{source['page']}")
