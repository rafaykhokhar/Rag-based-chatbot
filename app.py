import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os

# Load environment variables from .env file
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the OpenAI API key securely from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Spacy embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(tools, ques):
    # Initialize the language model using OpenAI's GPT-4 with the API key from environment
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5, api_key=openai_api_key)
    
    # Create a chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not available in the provided context, just say, 'answer is not available in the context'."""
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )
    
    # Create the tool calling agent
    agent = create_tool_calling_agent(llm, [tools], prompt)
    
    # Execute the agent with the user's question
    agent_executor = AgentExecutor(agent=agent, tools=[tools], verbose=True)
    response = agent_executor.invoke({"input": ques})
    
    # Display the response
    st.write("Reply: ", response['output'])

def user_input(user_question):
    # Load FAISS vector store
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    
    # Create a retriever tool for querying the PDF data
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(
        retriever, "pdf_extractor", "This tool is to give answers to queries from the PDF"
    )
    
    # Get conversational response using the retriever and the user's question
    get_conversational_chain(retrieval_chain, user_question)

def main():
    st.set_page_config("Chat PDF")
    st.header("RAG-based Chat with PDF")

    # User inputs a question to ask from the PDF
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If user inputs a question, process it
    if user_question:
        user_input(user_question)

    # Sidebar for uploading PDF files and processing them
    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process the uploaded PDF(s)
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Processing Complete!")

if __name__ == "__main__":
    main()
