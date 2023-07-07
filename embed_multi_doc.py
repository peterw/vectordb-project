import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma

PERSIST_DIRECTORY = 'db'

load_dotenv()

def split_documents(docs):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def load_and_split_documents():
    documents=[]
    loader = TextLoader(os.path.join('./content.txt'), encoding='utf-8')
    loaded_docs = loader.load_and_split()
    print(loaded_docs)
    documents.extend(loaded_docs)
    return documents

def embed_documents(documents):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()

def main():
    documents = load_and_split_documents()
    embed_documents(documents)
    print("Scrape Successful")

if __name__ == "__main__":
    main()