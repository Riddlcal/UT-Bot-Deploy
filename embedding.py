from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import dotenv
import csv
import os
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
