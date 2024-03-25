from flask import Flask, render_template, request
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.prompts.prompt import PromptTemplate
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunked_documents = []
for doc in documents:
    chunked_documents.extend(text_splitter.split_documents([doc]))

# Initialize OpenAI Embeddings
openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = 'text-embedding-3-small'
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model_name=model_name)

# Initialize Pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Now do stuff with Pinecone
index_name = "chatdata"

# Initialize Pinecone with the provided information
pc = Pinecone(api_key=pinecone_api_key, cloud="GCP", environment="gcp-starter", region="us-central1")

# Check if the index already exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,  # Length of OpenAI embeddings
        metric='cosine',  # or any other metric you prefer
        spec={"pod": "starter"}  # specify the correct variant and environment
    )

# Create a BM25Retriever instance with documents
retriever = BM25Retriever.from_documents(chunked_documents, k=2)

# Initialize Chat models
llm_name = 'gpt-4-1106-preview'
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(openai_api_key=openai_api_key, model=llm_name),
    retriever,
    return_source_documents=True
)

# Define the prompt template
prompt_template = """
You are a chatbot that answers questions about University of Texas at Tyler.
You will answer questions from students, teachers, and staff. Also give helpful hyperlinks to the relevant information.
If you don't know the answer, say simply that you cannot help with the question and advise to contact the host directly.

{question}
"""

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
    
        # Find all URLs and email addresses in the text
        urls = re.findall(r'\bhttps?://\S+\b', str(soup))
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', str(soup))
        
        # Replace each URL with an anchor tag
        for url in urls:
            # Create a new anchor tag
            new_tag = soup.new_tag('a', href=url, target='_blank', rel='noopener noreferrer')
            # Append icon to the link
            icon_tag = BeautifulSoup('<i class="fa-solid fa-arrow-up-right-from-square" style="margin-left: 10px;"></i>', 'html.parser')
            new_tag.append('Click here ')
            new_tag.append(icon_tag)
            # Replace the URL with the anchor tag
            soup = BeautifulSoup(str(soup).replace(url, str(new_tag)), 'html.parser')
        
        # Replace each email address with a mailto link
        for email in emails:
            # Create a new anchor tag
            new_tag = soup.new_tag('a', href='mailto:' + email)
            new_tag.append('Contact ')
            # Append icon to the link
            icon_tag = BeautifulSoup('<i class="fa-solid fa-envelope" style="margin-left: 10px;"></i>', 'html.parser')
            new_tag.append(icon_tag)
            # Replace the email with the anchor tag
            email_tag_str = str(new_tag)
            soup = BeautifulSoup(str(soup).replace(email, email_tag_str), 'html.parser')
        
        # Convert back to string and remove any loose characters after links
        answer_with_links = str(soup).strip().rstrip('/. ')
        
        # Add line breaks
        answer_with_line_breaks = answer_with_links.replace('\n', '<br>')
        
        # Check if bullet points are needed
        if '•' in answer_with_line_breaks:
            # Split the answer into lines and wrap each line with <li> tags
            lines = answer_with_line_breaks.split('\n')
            bulleted_lines = '<ul>' + ''.join([f'<li>{line}</li>' for line in lines]) + '</ul>'
            return bulleted_lines
        else:
            return answer_with_line_breaks

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
