import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone()

## Download and save embedding model and configs locally 
def get_embeddings(EMBEDDINGS_PATH, model_name):
    if not os.path.exists(EMBEDDINGS_PATH):
        print("Downloading HuggingFace embeddings...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer.save_pretrained(EMBEDDINGS_PATH)
        model.save_pretrained(EMBEDDINGS_PATH)
    else:
        print("Loading HuggingFace embeddings from disk...")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDINGS_PATH)
        model = AutoModel.from_pretrained(EMBEDDINGS_PATH)
    return HuggingFaceBgeEmbeddings(cache_folder=EMBEDDINGS_PATH, model_name=model_name)



## create pinecone index
def create_pc_index(index_name,dimension=1024):   
    pc = Pinecone()
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(    
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 
        

## make pinecone vectorstore object and add docs to the index
def create_vectorstore(PDFs_folder_path,index_name,embeddings):
    print("Adding Docs to PINECONE index...")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    pdf_files = [f for f in os.listdir(PDFs_folder_path)]
    loaders = [PDFPlumberLoader(os.path.join(PDFs_folder_path, file)) for file in pdf_files]
    docs = []
    count = 0
    for loader in loaders:
        count+=1
        print(f"Adding doc# {count}")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(docs[:20])
        vectorstore.add_documents(docs)
    return vectorstore
