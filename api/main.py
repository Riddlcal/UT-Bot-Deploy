from flask import Flask, render_template, request
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import re
import sys
import ctypes
import os

# Get the absolute path to the root directory
root_directory = os.path.abspath(os.path.dirname(__file__))

# Construct the path to the directory containing faiss_cpu_packages
faiss_path = os.path.join(root_directory, 'faiss_cpu_packages')

# Construct the path to the directory containing DLL files
faiss_libs_path = os.path.join(faiss_path, 'faiss_cpu.libs')

# Load all Faiss-related DLL files manually
dll_files = [
    'openblas-1ba25ee8d70fa3c45ede15bdc95fbee3.dll',
    'flang-d38962844214aa9b06fc3989f9adae5b.dll',
    'flangrti-5bbaf6aff159e72f9b015d5bc31c7584.dll',
    'libomp140.x86_64-21fc660a28479b04d9fec85174fc894e.dll',
    'libomp-26259cba665d756e8627cb8a206937cd.dll',
    'msvcp140-75f0e17aa84445df8d9f6c5be7aa31ac.dll'
]

for dll_file in dll_files:
    dll_path = os.path.join(faiss_libs_path, dll_file)
    ctypes.CDLL(dll_path)

# Add the directory containing Faiss to the PATH environment variable
os.environ['PATH'] = faiss_path + os.pathsep + os.environ['PATH']

sys.path.append("faiss_cpu_packages")

import faiss

app = Flask(__name__)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    return raw_text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_storage

def start_conversation(vector_embeddings):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    system_template = """You are a chatbot that answers questions about University of Texas at Tyler.
    You will answer questions from students, teachers, and staff. If you don't know the answer, say simply that you cannot help with the question and advise to contact the host directly.
    ----------------
    {context}"""

    human_template = "{question}"

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_embeddings.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )

    return conversation

# Read content from UT Bot.txt
file_path = "UT Bot.txt"
text_content = read_text_file(file_path)

# Process text content
chunks = get_chunks(text_content)
vector_embeddings = get_embeddings(chunks)
conversation = start_conversation(vector_embeddings)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Sample route for handling POST requests
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    response = conversation.invoke(user_input)
    bot_response = response['answer']

    # Remove labels like '[Label]' from the answer
    answer = re.sub(r'\[[^\[\]]+\]', '', bot_response)

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
