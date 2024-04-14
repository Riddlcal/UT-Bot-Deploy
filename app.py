from flask import Flask, render_template, request
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from bs4 import BeautifulSoup
import dotenv
import csv
import os
import re
import sys
maxInt = sys.maxsize

app = Flask(__name__)

dotenv.load_dotenv()

# DATA PROCESSING
columns_to_embed = ['url','text']
columns_to_metadata = ["url","text","date"]

docs = []
with open('UT Bot.csv', newline='', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata =to_metadata)
        docs.append(newDoc)

splitter = CharacterTextSplitter(separator="\n",
                                 chunk_size = 8000,
                                 chunk_overlap = 0,
                                 length_function =len)
documents = splitter.split_documents(docs)

#DATA EMBEDDING

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", show_progress_bar=True)

db = Chroma.from_documents(documents, embeddings)

# CHATBOT SET UP
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
openai_api_key = os.getenv("OPENAI_API_KEY")

general_system_template = r"""
You are a chatbot that answers questions about University of Texas at Tyler.
You will answer questions from students, teachers, and staff. If you don't know the answer, say simply that you cannot help with the question and advise to contact the host directly.
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
    bot_response = result['answer']

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
