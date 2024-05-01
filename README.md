# UT-Bot
A RAG chatbot for the University of Texas at Tyler - Capstone Project

We built a RAG (Retrieval-Augmented Generation) chatbot for the University of Texas at Tyler.

How does it work?

![rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a](https://github.com/Riddlcal/UT-Bot/assets/127689960/6c712ef6-687c-4668-881d-261f2cf850ac)

First we create embeddings of the text and place them into a vector database (like ChromaDB).

![rag_indexing-8160f90a90a33253d0154659cf7d453f](https://github.com/Riddlcal/UT-Bot/assets/127689960/6f951cdc-97b9-4c21-8116-9769be552936)

Then when a user asks a question, UT Bot finds the most relevant answer to the query and returns it.

app.py - for Render Deployment
bot.py - for local running

# How to run UT Bot

Clone the GitHub repo: https://github.com/Riddlcal/UT-Bot-Deploy

Open Command Prompt

Type and run:
```bash
git clone https://github.com/Riddlcal/UT-Bot-Deploy
```

Wait for cloning to finish, and then type and run:
```bash
cd UT-Bot-Deploy
```

Then to install dependencies, type and run:
```bash
pip install -r .\requirements.txt
```

After dependencies are installed, type and run:
```bash
python bot.py
```

Wait for Flask to start and open the localhost link
