import os
import subprocess

def install_chroma():
    # Install Chroma using pip
    subprocess.run(["pip", "install", "chroma"])

def handler(event, context):
    # Check if the function should install Chroma
    if event.get('install_chroma', False):
        install_chroma()
        return {"message": "Chroma installed successfully"}
    else:
        return {"message": "No action required"}