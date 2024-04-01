from flask import Flask, render_template, request
from openai import OpenAI
import sqlite3
import time
import re
from bs4 import BeautifulSoup

app = Flask(__name__)
client = OpenAI()

# Initialize SQLite database
conn = sqlite3.connect('threads.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS threads
             (id TEXT)''')
conn.commit()

# Retrieve file
def retrieve_file(file_id):
    file = client.files.retrieve(file_id)
    return file

file_id = "file-aEk13iLb3GpYB7ugAddJ7v34"
file = retrieve_file(file_id)

# Get or create assistant
def get_or_create_assistant(file):
    assistant_id = "asst_hZZ6WJuvudWehHO3Bmdsn2Ro"
    try:
        assistant = client.beta.assistants.retrieve(assistant_id)
    except Exception as e:
        # Assistant doesn't exist, create a new one
        assistant = client.beta.assistants.create(
            name="UTY",
            instructions="You are a chatbot that answers questions about University of Texas at Tyler. You will answer questions from students, teachers, and staff. Also give helpful hyperlinks to the relevant information. If you don't know the answer, and advise to contact the host directly. Keep answers short.",
            tools=[{"type": "retrieval"}],
            model="gpt-3.5-turbo-0125",
            file_ids=[file.id],
        )
    return assistant

assistant = get_or_create_assistant(file)

# Thread management
def check_if_thread_exists():
    c.execute('SELECT id FROM threads')
    thread_id = c.fetchone()
    return thread_id[0] if thread_id else None

def store_thread(thread_id):
    c.execute("INSERT INTO threads (id) VALUES (?)", (thread_id,))
    conn.commit()

# Generate response
def generate_response(message_body):
    # Check if there is already a thread_id for the current thread
    thread_id = check_if_thread_exists()

    # If a thread doesn't exist, create one and store it
    if thread_id is None:
        print("Creating new thread.")
        thread = client.beta.threads.create()
        store_thread(thread.id)
        thread_id = thread.id

    # Otherwise, retrieve the existing thread
    else:
        thread = client.beta.threads.retrieve(thread_id)

    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_body,
    )

    # Run the assistant and get the new message
    new_message = run_assistant(thread, assistant.id)
    return new_message

# Run assistant
def run_assistant(thread, assistant_id):
    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    # Wait for completion
    while run.status != "completed":
        # Be nice to the API
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # Retrieve the Messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    new_message = messages.data[0].content[0].text.value
    return new_message

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    response = generate_response(question)

    # Remove labels like '[Label]' from the answer
    answer = re.sub(r'\[[^\[\]]+\]', '', response)

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

if __name__ == '__main__':
    app.run(debug=True)
