import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()

pinecone.init(
    api_key='0d419af5-7392-449b-a5b0-0336c89e4a90',  # find at app.pinecone.io
    environment='eu-west1-gcp' #os.environ.get('PINECONE_ENV'), # next to api key in console
)

index_name = "pinecone-store-demo"

def split_documents(docs):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def load_and_split_documents(doc_path):
    loader = PyPDFLoader(doc_path)
    pages = loader.load_and_split()
    return pages

def embed_documents(documents):
    embedding = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embedding, index_name=index_name)
    
def main():
    documents_paths = ["intercom_1.pdf", "intercom_2.pdf"]
    for doc_path in documents_paths:
        documents = load_and_split_documents(doc_path)
        embed_documents(documents)
    print("Scrape Successful")

if __name__ == "__main__":
    main()
