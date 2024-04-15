from flask import Flask, render_template, request
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, 
                               SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate)
from bs4 import BeautifulSoup
import dotenv
import os
import re
import time
import warnings
import sys
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
maxInt = sys.maxsize
# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
app = Flask(__name__)


dotenv.load_dotenv()



CHROMA_PATH = 'chroma_data'

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(),
)


# CHATBOT SET UP
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
openai_api_key = os.getenv("OPENAI_API_KEY")

general_system_template = r"""
You are an AI chatbot named UT BOT.


Your primary goal is to provide accurate information about the Human Resources Development at University of Texas at Tyler provided the context.



---- {context} ----

"""

general_user_template = "Question:```{question}```"

messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]

prompt = ChatPromptTemplate.from_messages(messages=messages)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key,model_name="gpt-3.5-turbo-0125")


qa = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    memory = memory,
    combine_docs_chain_kwargs={'prompt': prompt},
    chain_type="stuff",
    return_source_documents=True,
)  

## RUN THE APPLICATION
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    result = qa.invoke({"question": user_input, "chat_history": {}})
    answer = result['answer']

    time.sleep(0.5)

    # Create a BeautifulSoup object
    soup = BeautifulSoup(answer, 'html.parser')

    # Handle markdown links
    markdown_links = re.findall(r'\[(.+?)\]\((.+?)\)', str(soup))
    for link_text, link_url in markdown_links:
        link_tag = soup.new_tag('a', href=link_url, target='_blank', rel='noopener noreferrer')
        link_tag.string = link_text
        icon = soup.new_tag('i', attrs={'class': 'fa-solid fa-arrow-up-right-from-square', 'style': 'margin-left: 10px;'})
        link_tag.append(icon)
        soup.append(link_tag)
        
            # Replace the markdown link with an empty string
        answer = answer.replace(f'[{link_text}]({link_url})', '')

        # Handle HTML links
    for link in soup.find_all('a'):
        link['target'] = '_blank'
        link['rel'] = 'noopener noreferrer'
        icon = soup.new_tag('i', attrs={'class': 'fa-solid fa-arrow-up-right-from-square', 'style': 'margin-left: 10px;'})
        link.append(icon)

    # Get the formatted answer with webpage links
    answer_with_webpage_links = str(soup)

        # Handle email addresses as links using regular expressions
    answer_with_links = re.sub(r'(\S+@\S+)', r'<a href="mailto:\1">Contact<i class="fa-solid fa-envelope" style="margin-left: 10px;"></i></a>', answer_with_webpage_links)

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

if __name__ == '__main__':
    app.run(debug=True)
