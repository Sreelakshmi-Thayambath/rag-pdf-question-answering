import streamlit as st
import requests

st.set_page_config(page_title="PDF QA System")

st.title("📄 PDF Question-Answering System")

question = st.text_input("Ask a question from the PDF")

if st.button("Get Answer"):

    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"question": question}
    )

    if response.status_code == 200:
        answer = response.json()["answer"]

        st.subheader("Answer")
        st.write(answer)

    else:
        st.error("Error connecting to API")