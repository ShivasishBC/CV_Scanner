import streamlit as st
import openai
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import docx2txt
from pypdf import PdfReader

load_dotenv()

# Langsmith tracking
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_PROJECT'] = "CV Screening Model"

# Prompt Template for scoring CVs
# prompt_template = """
# You are a helpful assistant that scores resumes based on the provided job description (JD).
# Job Description: {jd}
# Resume: {resume}
# How relevant is this resume to the job description? Provide a relevance score between 0 and 100.
# Additionally, provide a point brief explanation for the given score, highlighting the key points that influenced the score.
# Format the response as follows:
# 1.Relevance Score: [score]
# 2.Explanation: [brief explanation]
# """

prompt_template = """
You are a helpful assistant that scores resumes based on the provided job description (JD).
Evaluate the resume based on the following criteria:
1. Ovarall Relevance Score (0-100 points): Ovarall score based on entire evaluation done
2. Skills Match (0-30 points): How well do the skills listed in the resume match the skills required in the JD?
3. Experience Match (0-30 points): How well does the candidate's experience match the job requirements?
4. Education Match (0-20 points): How well does the candidate's education align with the job requirements?
5. Overall Relevance (0-20 points): Overall relevance of the candidate's profile to the job.

Job Description: {jd}
Resume: {resume}

Provide a detailed score breakdown and a total relevance score out of 100.
Format the response as follows:
Relevance Score: [score]
Skills Match: [score]
Experience Match: [score]
Education Match: [score]
Overall Relevance: [score]
Total Relevance Score: [total score]
Explanation: [brief explanation for the scoring]
"""
def extract_text_from_file(file):
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    else:
        return file.read().decode("utf-8")

def generate_response(jd, resume, api_key, llm, temperature, max_tokens):
    prompt = prompt_template.format(jd=jd, resume=resume)
    client = OpenAI(api_key = api_key)
    
    response = client.chat.completions.create(
        model=llm,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# Streamlit App
st.title('Enhanced CV Screening Model')
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Your API Key Here", type='password')

# Dropdown to select Openai Model
llm = st.sidebar.selectbox("Select Openai Model", ["gpt-4", "gpt-4-turbo", "gpt-4o"])

# Adjust parameter using slider
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value=50, max_value=500, value=150)

# Main Interface for user Input
st.write("Upload Job Description and CVs to get relevance scores")

# Job Description input
jd_input = st.text_area('Job Description:')

# CV Upload
uploaded_files = st.file_uploader("Upload CVs", accept_multiple_files=True)

# Add a submission button
if st.button("Submit"):
    if jd_input and uploaded_files:
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_file(uploaded_file)
            response = generate_response(jd_input, resume_text, api_key, llm, temperature, max_tokens)
            st.write(f"CV: {uploaded_file.name}")
            st.write(response)
    else:
        st.write("Please provide the Job Description and upload CVs.")
