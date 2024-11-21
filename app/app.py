import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch
from peft import PeftModel
import PyPDF2  # For handling PDF files
import nltk
import re
from nltk.tokenize import sent_tokenize

# Download the necessary NLTK data (if not already installed)
nltk.download('punkt')

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model_path = "G:/SEM 7/NN/New_Project/models/QA_Peft"
    base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

Ques_Ans = []

# Function to generate Q&A based on input context
def model_generate(context_text):
    input_text = f"Generate Question and Answer or Fill in the Blanks from the given context: {context_text}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = {key: value.to(device) for key, value in input_ids.items()}
    # Generate output from the model
    output = model.generate(**input_ids, max_length=128, num_beams=5, early_stopping=True)

    # Decode the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)  
    return decoded_output

# Function to clean input text
def clean_input(text):
    """Clean input text for question generation."""
    text = text.strip()
    if len(text.split()) < 3:  # Skip if the text has fewer than 3 words
        return ""
    return text

# Function to generate Q&A from sentences
def generate_different_Q_A(input_text):
    """Generate Question Answers from input text for each sentence."""
    sentences = split_text_into_sentences(input_text)
    Q_A = []
    question_counter = 1  # Counter for numbering questions
    for sentence in sentences:
        cleaned_sentence = clean_input(sentence)
        if not cleaned_sentence:  # Skip invalid or too short sentences
            continue
        question_answer = model_generate(cleaned_sentence)
        
        # Split the model's output into question and answer parts
        if "Answer:" in question_answer:
            question, answer = question_answer.split("Answer:", 1)
            question = question.strip().replace("Question:", "").strip()
            answer = answer.strip()
        else:
            question = question_answer.strip()
            answer = "Not provided"  # Fallback if model output is incomplete

        # Add HTML styling for background color
        formatted_output = f"""
        <div style="background-color: #b5b5b5; color:black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>Question {question_counter}:</strong> {question}<br>
            <strong>Answer:</strong> {answer}
        </div>
        """
        Q_A.append((question, answer))  # Store question and answer for later use
        Ques_Ans.append(formatted_output)
        
        question_counter += 1
    return Ques_Ans, Q_A

# Function to split text into consecutive sentence pairs
def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentence_pairs = []
    for i in range(0, len(sentences)-1, 2):  
        pair = sentences[i] + " " + sentences[i+1]  
        sentence_pairs.append(pair)
    if len(sentences) % 2 != 0:
        sentence_pairs.append(sentences[-1])
    return sentence_pairs

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_pdf):
    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to read the content of a TXT file
def extract_text_from_txt(uploaded_txt):
    return uploaded_txt.read().decode("utf-8")

# Function to summarize text using BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    text_for = f"Summarize the following text into bullet points:\n\n{text}"
    result = summarizer(text_for, max_length=256, min_length=30, do_sample=False)
    return result[0]['summary_text']

# Function to save the Q&A to a text file
def save_qa_to_txt(q_a_list):
    file_content = ""
    for i, (question, answer) in enumerate(q_a_list):
        file_content += f"Question {i+1}:\n{question}\nAnswer {i+1}:\n{answer}\n\n"
    
    # Save to a .txt file
    with open("Result.txt", "w") as file:
        file.write(file_content)
    
    return "Result.txt"

# Streamlit App UI
def main():
    # st.title("AnswerCraft")
    # st.markdown("#### :")
    st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">AnswerCraft</h1>
    <h3 style="text-align: center; color: #757575;">Your one-stop solution for Q&A generation and text summarization</h3>
""", unsafe_allow_html=True)

    option_type = st.radio("Choose input:", ("Summarization", "Question Answer Generation"))

    if option_type == "Summarization":
        st.markdown("""
        <h4 style="text-align: center; color: #757575;">Summarize the given input:</h4>
        """, unsafe_allow_html=True)
        # st.markdown("### Summarize the given text:")
        option = st.radio("Choose input method:", ("Text Input", "Upload File"))

        if option == "Text Input":
            context_text = st.text_area("Input Context", height=100)
            if st.button("Generate Summary"):
                if context_text.strip():
                    with st.spinner("Generating Summary..."):
                        summary = summarize_text(context_text)
                        st.success("Summary:")
                        # Styling the summary output with dark background and light text
                        bullet_points = summary.split(". ")
                        for point in bullet_points:
                            st.markdown(
                                f'<div style="background-color: #b5b5b5; color:black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">- {point.strip()}</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.warning("Please enter some context text.")

        elif option == "Upload File":
            uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    context_text = extract_text_from_pdf(uploaded_file)
                    st.markdown("""
        <h4 style="text-align: center; color: #757575;">Content from Your File:</h4>
        """, unsafe_allow_html=True)
                    with st.expander("Click to view full content", expanded=False):
                        st.write(context_text)
                elif uploaded_file.type == "text/plain":
                    context_text = extract_text_from_txt(uploaded_file)
                    st.markdown("""
        <h4 style="text-align: center; color: #757575;">Content from Your File:</h4>
        """, unsafe_allow_html=True)
                    with st.expander("Click to view full content", expanded=False):
                        st.write(context_text)

                if st.button("Generate Summary"):
                    if context_text.strip():
                        with st.spinner("Generating Summary..."):
                            summary = summarize_text(context_text)
                            st.success("Summary:")
                            bullet_points = summary.split(". ")
                            for point in bullet_points:
                                st.markdown(
                                    f'<div style="background-color: #b5b5b5; color:black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">- {point.strip()}</div>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.warning("The uploaded file is empty or invalid.")

    elif option_type == "Question Answer Generation":
        st.markdown("""
        <h4 style="text-align: center; color: #757575;">Question Answer Generation from given Input:</h4>
        """, unsafe_allow_html=True)
        # Option to upload a file or input text
        option = st.radio("Choose input method:", ("Text Input", "Upload File"))

        if option == "Text Input":
            context_text = st.text_area("Input Context", height=100)
            if st.button("Generate Multiple Q&A"):
                if context_text.strip():
                    with st.spinner("Generating Q&A..."):
                        q_a_list, q_a_text = generate_different_Q_A(context_text)
                        st.success("Generated Q&A:")
                        for qa in q_a_list:
                            st.write(qa, unsafe_allow_html=True)
                        # Save Q&A to file
                        file_name = save_qa_to_txt(q_a_text)
                        st.download_button(
                            label="Download Q&A",
                            data=open(file_name, "rb").read(),
                            file_name=file_name,
                            mime="text/plain"
                        )
                else:
                    st.warning("Please enter some context text.")

        elif option == "Upload File":
            uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    context_text = extract_text_from_pdf(uploaded_file)
                    # st.subheader("Content from Your File:")
                    st.markdown("""
        <h4 style="text-align: center; color: #757575;">Content from Your File:</h4>
        """, unsafe_allow_html=True)
                    with st.expander("Click to view full content", expanded=False):
                        st.write(context_text)
                elif uploaded_file.type == "text/plain":
                    context_text = extract_text_from_txt(uploaded_file)
                    st.markdown("""
        <h4 style="text-align: center; color: #757575;">Content from Your File:</h4>
        """, unsafe_allow_html=True)
                    with st.expander("Click to view full content", expanded=False):
                        st.write(context_text)

                if st.button("Generate Multiple Q&A"):
                    if context_text.strip():
                        with st.spinner("Generating Q&A..."):
                            q_a_list, q_a_text = generate_different_Q_A(context_text)
                            st.success("Generated Q&A:")
                            for qa in q_a_list:
                                st.write(qa, unsafe_allow_html=True)
                            # Save Q&A to file
                            file_name = save_qa_to_txt(q_a_text)
                            st.download_button(
                                label="Download Q&A",
                                data=open(file_name, "rb").read(),
                                file_name=file_name,
                                mime="text/plain"
                            )
                    else:
                        st.warning("The uploaded file is empty or invalid.")

if __name__ == "__main__":
    main()
