from flask import Flask, render_template, request
from openai import OpenAI
import shelve
import time

app = Flask(__name__)
client = OpenAI()

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
            name="UT Bot",
            instructions="You are a chatbot that answers questions about University of Texas at Tyler. You will answer questions from students, teachers, and staff. Also give helpful hyperlinks to the relevant information. If you don't know the answer, say simply that you cannot help with the question and advise to contact the host directly.",
            tools=[{"type": "retrieval"}],
            model="gpt-4-1106-preview",
            file_ids=[file.id],
        )
    return assistant

assistant = get_or_create_assistant(file)

# Thread management
def check_if_thread_exists():
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get("current_thread", None)

def store_thread(thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf["current_thread"] = thread_id

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
    return response

if __name__ == '__main__':
    app.run(debug=True)
