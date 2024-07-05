import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.helper import get_embeddings, create_pc_index, create_vectorstore
from src.prompt import prompt
from pinecone import Pinecone
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()

## load the GROQ KEY
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt = prompt

# Download the embeddings and store them locally so
# next time they can be loaded directly from the local disk
# You can choose any other embedding model of your choice 
EMBEDDINGS_PATH = "./huggingface_bge_embeddings"
model_name = "BAAI/bge-large-en"
embeddings = get_embeddings(EMBEDDINGS_PATH, model_name)

# Create the index in Pinecone Vector DB
# If the index already exist, you can add more docs to it
# or simply use the docs already in the index
# For creating new index, uncomment the next line
# create_pc_index(index_name,dimension=1024)  

# If you want to use already existing index 
PDFs_folder_path = "data/"
index_name = "genai-library"
pc = Pinecone()
vectorstore = create_vectorstore(PDFs_folder_path, index_name, embeddings)


# Create retreival chain
def create_retreival_chain():
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vectorstore.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    return retrieval_chain

retrieval_chain = create_retreival_chain()


## creating the Flask app for Chatbot
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    input = request.form["msg"]
    print(input)
    response=retrieval_chain.invoke({'input':input})    
    answer = response['answer']
    print(answer)
    return str(answer)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= False)
