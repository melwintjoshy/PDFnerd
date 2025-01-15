from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
import google.generativeai as genai
import streamlit as st 
import os
from dotenv import load_dotenv 

load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key = st.secrets["GOOGLE_API_KEY"])

def loadpdf(uploaded_pdf):
    with open("temp_uploaded_pdf.pdf", "wb") as temp_file:
                temp_file.write(uploaded_pdf.read())
                loader = PyPDFLoader("temp_uploaded_pdf.pdf")
                text_documents = loader.load()
    return text_documents

def getchunks(text_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(text_documents)
    return chunks
                    
def embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings( model="models/embedding-001")
    db = FAISS.from_documents(chunks, embedding = embeddings)
    db.save_local("db")

def chain():
    #llm model
    llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    #designing prompt
    prompt =  ChatPromptTemplate.from_template(
        """You are an assistant to interact with pdf. Understand the context of the document.
        Answer the following query based on the provided context. If the question is unclear or not directly related to the document, very politely ask the user for clarification or inform them.
        Summarize the context if asked.
        {context}
        </context>
        Question : {input}"""
    )

    #chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

def user_input(input_text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("db", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    document_chain = chain()

    #rag chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)


    if input_text:
        response = retrieval_chain.invoke({"input":input_text})
        return(response['answer'])

def clear_chat_history():
    st.session_state.messages = [
        {"role": "PDFnerd", "content": "Upload your PDF and ask me your questions."}]

def main():
    st.set_page_config(page_title="PDFnerd by mxlwin", layout="wide")
    st.title("What do you want to know?")

    with st.sidebar:
        custom_title = """
                        <h1 style="font-size: 50px; color: white; font-family: 'Helvetica', sans-serif;">PDFnerd</h1>
                       """
        st.markdown(custom_title, unsafe_allow_html=True)
        st.write("Simplify, Search, Solve: Your PDF Chat Companion!")
        st.write("")

        uploaded_pdf = st.file_uploader("Upload your PDF file:", type="pdf")

        if st.button("Submit"):
            if uploaded_pdf is not None:
                with st.spinner("Processing..."):
                    text_documents = loadpdf(uploaded_pdf)        
                    chunks = getchunks(text_documents)     
                    embeddings(chunks)
                    st.success("I'm ready.")
                    
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    #default first message
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "PDFnerd", "content": "Upload your PDF and ask me your questions."}]

    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
           st.markdown(f"**{message['role']}:** {message['content']}")


    # input message 
    input_text = st.chat_input()
    if input_text:
        st.session_state.messages.append({"role": "User", "content": input_text})
        with st.chat_message("User"):
            st.markdown(f"**User:** {input_text}")
            
        #output message
        if st.session_state.messages[-1]["role"] != "PDFnerd":
            with st.chat_message("PDFnerd"):
                with st.spinner("Generating..."):
                    response = user_input(input_text)
                    st.markdown(f"**PDFnerd:** {response}")
            st.session_state.messages.append({"role": "PDFnerd", "content": response})        
                    

if __name__ == "__main__":
    main()
