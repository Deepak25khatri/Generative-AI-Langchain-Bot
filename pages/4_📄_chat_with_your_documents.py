import streamlit as st
from streamlit_lottie import st_lottie
import PyPDF2 as pdf
from langchain.text_splitter import CharacterTextSplitter as ct
from langchain.embeddings.cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Cohere
from chatbot_style import css, bot_template, user_template
import json
import os
from dotenv import load_dotenv
import utils

st.title("Chat with multiple PDFs 👋🏻")
st.write("You Must Ensure that you have uploaded the PDF files before asking a question")

class DocChatbot:

    def __init__(self):
        self.cohere_model = None

    # To persist the conversation context
    def get_conversation_chain(self, vector_store):
        llm = Cohere(model="command")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, human_prefix="Human", ai_prefix="Bot")
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
        )
        return conversation_chain

    def main(self):
        # Ensure API key is entered
        utils.configure_cohere()

        if "COHERE_API_KEY" in st.session_state:
            self.cohere_model = st.session_state["COHERE_API_KEY"]

            question = st.text_input("Hi! How May I Help?", key="user_question_input")

            # Initializing Session States
            if "conversation" not in st.session_state:
                st.session_state.conversation = None
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = None

            # Load CSS for chatbot styling
            st.write(css, unsafe_allow_html=True)

            if question and st.session_state.conversation:
                response = st.session_state.conversation({"question": question})
                st.session_state.chat_history = response["chat_history"]
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            with st.sidebar:
                st.header("Upload your PDFs 📚")
                doc_pdf_files = st.file_uploader("Select your PDF files & Click on 'Process' to proceed", accept_multiple_files=True, key="pdf_uploader")
                submit = st.button("Process", key="process_button")
                if submit:
                    with st.spinner("Processing! Wait a While...."):
                        # Extract PDF text using PyPDF2
                        text = ""
                        for file in doc_pdf_files:
                            reader = pdf.PdfReader(file)
                            for page_number in range(len(reader.pages)):
                                page = reader.pages[page_number]
                                text += str(page.extract_text())

                        # Chunk them up using langchain textsplitter
                        text_chunker = ct(
                            separator="\n",
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len,
                            add_start_index=True
                        )
                        text_chunks = text_chunker.split_text(text)
                                    
                        # Create Embeddings & Store them in the Vector Database
                        embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
                        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                        
                        if vector_store:
                            st.write("✅ File is uploaded & embedded, you may ask your questions now.")
                        else:
                            st.write("❗️ Failed to process the document.")
                        
                        # Create Chat Buffer Memory using langchain
                        st.session_state.conversation = self.get_conversation_chain(vector_store)

if __name__ == "__main__":
    obj = DocChatbot()
    obj.main()
