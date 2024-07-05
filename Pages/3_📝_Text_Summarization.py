import streamlit as st
from langchain_cohere import ChatCohere
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import utils
st.set_page_config(page_title='Text Summarization App', page_icon='📝')
st.title('Text Summarization App')

class TextSummarizer:
    def __init__(self):
        self.cohere_model = utils.configure_cohere()

    def generate_response(self, txt):
        # Instantiate the Cohere model
        llm = ChatCohere(model=self.cohere_model, temperature=0)
        # Split text
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(txt)
        # Create multiple documents
        docs = [Document(page_content=t) for t in texts]
        # Text summarization
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        return chain.run(docs)

    @utils.enable_chat_history
    def main(self):
       
        

        # Text input
        txt_input = st.text_area('Enter your text', '', height=200)

        # Form to accept user's text input for summarization
        if txt_input:
            utils.display_msg(txt_input, 'user')
            with st.chat_message("assistant"):
                with st.spinner('Summarizing...'):
                    response = self.generate_response(txt_input)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    summarizer = TextSummarizer()
    summarizer.main()