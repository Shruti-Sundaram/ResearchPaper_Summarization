import os
import streamlit as st
import base64
import fitz  
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("shruti28062000/bart_finetuned_papers_4e")
model = AutoModelForSeq2SeqLM.from_pretrained("shruti28062000/bart_finetuned_papers_4e")


model.to(device)


# Function to summarize text using finetuned model
def summarize(text, max_summary_length=700):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(
        inputs,
        max_length=max_summary_length,
        min_length=int(max_summary_length / 5),
        length_penalty=8.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to split text into pieces for summarization
def text_split(text, max_tokens=900, overlap_percent=15):
    tokens = tokenizer.tokenize(text)
    token_overlap = int(max_tokens * overlap_percent / 100)
    pieces = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - token_overlap)]
    text_pieces = [tokenizer.decode(tokenizer.convert_tokens_to_ids(piece), skip_special_tokens=True) for piece in pieces]
    return text_pieces

# Function to recursively summarize large text
def recursive_func(text, max_length=700, min_final_length=200, recursion_level=0):
    recursion_level += 1
    tokens = tokenizer.tokenize(text)
    expected_count_of_chunks = len(tokens) / max_length
    if expected_count_of_chunks < 1:
        # If the text is already short enough, summarize it directly
        return summarize(text, max_summary_length=max_length)

    max_chunk_length = int(len(tokens) / expected_count_of_chunks) + 2
    pieces = text_split(text, max_tokens=max_chunk_length)
    summaries = [summarize(piece, max_summary_length=max_chunk_length // 3 * 2) for piece in pieces]
    concatenated_summary = ' '.join(summaries)
    tokens = tokenizer.tokenize(concatenated_summary)

    if len(tokens) > max_length:
        return recursive_func(concatenated_summary, max_length=max_length, min_final_length=min_final_length, recursion_level=recursion_level)
    else:
        if len(tokens) < min_final_length:
            return summarize(concatenated_summary, max_summary_length=max_length)
        else:
            return concatenated_summary


def clean_summary(summary):
    cleaned_summary = re.sub(r'\s+', ' ', summary)  
    return cleaned_summary.strip()  

# Function to clean the output
def nltk_clean_text(text):
    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = re.sub(r'(?<=[.,;:])(?!\s)', r' ', text)
    text = re.sub(r'(\.)(\s*)([a-z])', lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text)
    text = re.sub(r'(:)(\s*)([a-z])', lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text)
    text = re.sub(r'\bIntroduction(?!:)', r'Introduction:', text)
    text = re.sub(r'\bAbstract(?!:)', r'Introduction:', text)
    text = re.sub(r'\bObjective\b(?!:)', r'Objective:', text)
    text = re.sub(r'\bConclusions(?!:)', r'Conclusions ', text)
    ttext = re.sub(r'\bConclusion(?![s])', r'Conclusion:', text)
    text = re.sub(r'\bResults(?!:)', r'Results:', text)
    text = re.sub(r'\bdiscussion(?!:)', r'discussion:', text)
    text = re.sub(r'\bobjectives(?!:)', r'objectives:', text)
    text = re.sub(r'\bmethods(?!:)', r'methods ', text)
    text = re.sub(r'(?<=:)(?!\s)', r' ', text)
    text = re.sub(r'\b[Hh]e\b', 'it', text)
    text = re.sub(r'\b[Ss]he\b', 'it', text)

    return text


# Function to convert PDF to Markdown 
def convert_pdf_to_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


def pdf_to_plain_text(pdf_path):
    markdown_text = convert_pdf_to_markdown(pdf_path)
    final_summary = recursive_func(markdown_text)
    final_summary = clean_summary(final_summary)
    final_summary = nltk_clean_text(final_summary)
    return final_summary

@st.cache_data
# Function to display the PDF in an iframe
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code 
st.set_page_config(layout="wide")

def main():
    st.markdown("""
    <h3 style='text-align: center; margin-bottom: 50px;'>Generate Summaries for Research Papers</h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)

    if uploaded_file:
        file_names = [file.name for file in uploaded_file]
        selected_file_name = st.selectbox("Select a PDF file", file_names)
        selected_file = next(file for file in uploaded_file if file.name == selected_file_name)

        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            output_dir = "sum_pdf"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, selected_file_name)
            with open(filepath, "wb") as temp_file:
                temp_file.write(selected_file.read())

            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)
                
            with col2:
                summary_placeholder = st.info("Generating Summary...")
                summary = pdf_to_plain_text(filepath)
                summary_placeholder.empty()
                st.info("Summary")
                st.success(summary)

if __name__ == "__main__":
    main()
