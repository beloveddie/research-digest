import streamlit as st
import pdfplumber
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Cohere API client
co = cohere.Client(os.environ["COHERE_API_KEY"])

# Text Extraction from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Summarize Text Using Cohere
def summarize_text(text, section="main points"):
    prompt = f"Summarize the following {section} of a research paper in 3-4 sentences: {text}"
    response = co.generate(
        model="command-xlarge-nightly",  # Use the appropriate model
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

# Question-Answering Using Cohere
def answer_question(text, question):
    prompt = f"Using the following text, answer the question accurately:\n\nText: {text}\n\nQuestion: {question}"
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

# Streamlit App
st.title("ResearchDigest: Multi-Paper Summarizer & QA System")
st.subheader("Upload research papers to summarize them and ask questions.")

uploaded_files = st.file_uploader("Upload Research Papers (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_texts = {}
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Extract text from each uploaded file
            pdf_text = extract_text_from_pdf(uploaded_file)
            all_texts[uploaded_file.name] = pdf_text
    
    # Display Summaries for Each Paper
    st.subheader("Summaries")
    for file_name, text in all_texts.items():
        with st.spinner(f"Summarizing {file_name}..."):
            summary = summarize_text(text[:1000])  # Summarize first 1000 characters
        st.write(f"**{file_name}**")
        st.write(summary)
    
    # Add QA System
    st.subheader("Ask Questions")
    selected_file = st.selectbox("Select a paper to ask questions about:", list(all_texts.keys()))
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Generating answer..."):
            answer = answer_question(all_texts[selected_file], question)
        st.write(f"**Answer:** {answer}")