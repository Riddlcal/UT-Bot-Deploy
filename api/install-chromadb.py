import subprocess

def install_chromadb():
    try:
        subprocess.run(["pip", "install", "chromadb"])
        return "ChromaDB installed successfully"
    except Exception as e:
        return f"Error installing ChromaDB: {str(e)}"

def handler(event, context):
    return {
        "statusCode": 200,
        "body": install_chromadb()
    }