from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables (in case you use them for LangSmith or others)
load_dotenv()

# Langsmith tracking (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("user", "Question: {question}")
])

# Streamlit interface
st.title("Langchain chatbot with Ollama (LLaMA 2)")
input_text = st.text_input("Search the topic you want:")

# LLM and Chain setup using Ollama with llama2 model
llm = Ollama(model="llama2")  # Make sure this model is pulled: `ollama pull llama2`
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Run the chain
if input_text:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({'question': input_text})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
