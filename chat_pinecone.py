import os
import re
import openai
import streamlit as st
import shutil
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_chat import message
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

INDEX_NAME = "pinecone-store-demo"
st.write("# Pinecone Chatbot")

@st.cache_resource
def load_pinecone_existing_index():
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    return docsearch

db=load_pinecone_existing_index()

def get_user_query():
    with st.form(key="my_form"):
        input_prompt = "Enter query"
        input = st.text_input(input_prompt, value="", key="input")
        submit_button = st.form_submit_button(label="Submit")
        if not input and submit_button:
            st.info("Input blank!")
    return input

def search_database(query):
    result_docs = db.similarity_search(query)
    print('result_docs' + str(result_docs))
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="refine", verbose=True)
    output = chain({"input_documents": result_docs, "question": query})

    return output['output_text']

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def main():
    user_query = get_user_query()   

    if user_query:
        output = search_database(user_query)
        st.session_state.past.append(user_query)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    main()
