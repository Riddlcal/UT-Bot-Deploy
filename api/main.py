import itertools
import asyncio
import random
from flask import Flask, render_template, request
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, PodSpec
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup
import time
import os
import warnings
import re

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)

# Specify the file path to UT Bot.txt
file_path = "UT Bot.txt"
loader = TextLoader(file_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents = []
for doc in documents:
    chunked_documents.extend(text_splitter.split_documents([doc]))

# Initialize APIs
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Use pod-based architecture
spec = PodSpec(environment='GCP')

# Initialize Pinecone with the provided information
pc = Pinecone(api_key=pinecone_api_key)

# Set index name
index_name = "chatdata"

# Check if the index already exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Length of OpenAI embeddings
        metric='cosine',  # or any other metric you prefer
        spec=spec  # specify the correct variant and environment
    )

# Wait for index to be initialized
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Initialize LangChain embeddings object
model_name = 'text-embedding-3-small'
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

def generate_example_data(vector_dim, vector_count):
    for i in range(vector_count):
        yield f'id-{i}', [random.random() for _ in range(vector_dim)]

# Define the vector dimension and count
vector_dim = 1536
vector_count = 10000

# Example generator that generates many (id, vector) pairs
example_data_generator = generate_example_data(vector_dim, vector_count)

# Define the chunks function to break an iterable into chunks
def chunks(iterable, batch_size=10):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

async def upsert_data_async(pc, index_name, example_data_generator, batch_size):
    index = pc.Index(index_name)
    for ids_vectors_chunk in chunks(example_data_generator, batch_size):
        await index.upsert(vectors=ids_vectors_chunk, async_req=True)

# Initialize Chat models
llm_name = 'gpt-3.5-turbo'
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(openai_api_key=openai_api_key, model=llm_name),
    None,  # No need for retriever
    return_source_documents=True
)

# Define the prompt template
prompt_template = """
You are a chatbot that answers questions about University of Texas at Tyler.
You will answer questions from students, teachers, and staff. Also give helpful hyperlinks to the relevant information.
If you don't know the answer, say simply that you cannot help with the question and advise to contact the host directly.

{question}
"""

# Initialize the LangChain vector store
text_field = "text"
vectorstore = PineconeVectorStore(index_name, embeddings, text_field)

# Initialize RetrievalQA object
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=llm_name, temperature=0.0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Sample route for handling POST requests
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    result = qa.invoke({"question": user_input, "chat_history": {}})
    answer = result['answer']

    # Sleepy time
    time.sleep(0.5)

    # Remove labels like '[Label]' from the answer
    answer = re.sub(r'\[[^\[\]]+\]', '', answer)

    # Check if the answer contains iframe HTML
    if 'iframe' in answer:
        return render_template('iframe.html', iframe_html=answer)
    else:
        # Remove unnecessary characters such as parentheses
        cleaned_answer = re.sub(r'[()\[\]]', '', answer)

        # Initialize Beautiful Soup
        soup = BeautifulSoup(cleaned_answer, 'html.parser')

