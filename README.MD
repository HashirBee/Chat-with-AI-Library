
# Chat with AI Library

_Interact with a talking AI Library through this GenAI chatbot !!!_

This repository contains the code for Talking AI Library, a GenAI chatbot, designed as a GenAI project that can be a starting point and an indicator of something really useful.
The idea is that I used Langchain framework and Pinecone vectorstore to create a RAG-powered LLM that answers the questions related to AI especially GenAI. It uses some of the most famous books as a context. I uploaded only one book to show an instance. Locally, I used more than 10 books to develop the LLM app.

I have used Groq's Fast AI inference, utilizing Meta's opensource Llama 3 LLM to build this RAG chatbot.


## Features

- Talk to an AI library
- Chat with authentic and selected books 
- Quick access to the desired information
- Summarized/ Concise information


## Tech Stack

- [LangChain](https://www.langchain.com/)
- [HuggingFace Embeddings](https://ollama.com/)
- [Pinecone](https://www.pinecone.io/)
- [Groq Inference Engine](https://wow.groq.com/why-groq/)


## Books I used
[Google Drive](https://drive.google.com/drive/folders/16t0mG5ZOtTdmPY-Mn9WrmO5IAfCpv64N?usp=sharing)


## Deployment Instructions


Due to insufficient computing resources, I only used first few documents from each book in the vector DB. If you want to make a large-scale professional app, please insert all the required books fully into vector DB and also go on to fine-tuning.

To deploy this project run, clone the repository, get API keys for Groq, HuggingFace and Pinecone, pip install requirements.txt and run following command in cmd

```bash
  python app.py
```

