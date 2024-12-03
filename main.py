import streamlit as st
import pdfplumber
import cohere
import re
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
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
        model="command-xlarge",  # Use the appropriate model
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

# Extract Citations
def extract_citations(text):
    citation_pattern = r'\[(.*?)\]'  # Regex for citations (e.g., [1] or [Smith et al., 2020])
    citations = re.findall(citation_pattern, text)
    return citations

# Build and Visualize Citation Network
def build_citation_network(citations):
    G = nx.Graph()
    
    for citation in citations:
        G.add_node(citation)
    
    for i, citation in enumerate(citations[:-1]):
        G.add_edge(citation, citations[i+1])
    
    # Plot the network
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
    plt.title("Citation Network")
    return plt

# Streamlit App
st.title("ResearchDigest: Paper Summarizer & Citation Network Analyzer")
st.subheader("Upload a research paper to get a summary and analyze its citation network.")

uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Extract text from uploaded PDF
    with st.spinner("Extracting text from the PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Summarize the text
    with st.spinner("Summarizing the research paper..."):
        summary = summarize_text(pdf_text[:1000])  # Process first 1000 characters for demo
    
    # Display summary
    st.subheader("Summary")
    st.write(summary)
    
    # Extract and visualize citations
    with st.spinner("Analyzing citations..."):
        citations = extract_citations(pdf_text)
        st.subheader("Citations")
        st.write(citations)
        
        if citations:
            st.subheader("Citation Network")
            fig = build_citation_network(citations)
            st.pyplot(fig)
        else:
            st.write("No citations detected.")