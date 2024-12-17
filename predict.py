# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import json
import time
import base64
import requests
import subprocess
from cog import BasePredictor, Input, Path, ConcatenateIterator

MODEL_NAME = "llama3.2-vision:90b"
OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_CHAT = OLLAMA_API + "/api/chat"
MODEL_CACHE = "checkpoints"
# MODEL_URL = "https://weights.replicate.delivery/default/ollama/llama3.2-vision/11b.tar"
MODEL_URL = "https://weights.replicate.delivery/default/ollama/llama3.2-vision/90b.tar"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)
    
def wait_for_ollama(timeout=120):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(OLLAMA_API)
            if response.status_code == 200:
                print("Ollama server is running")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("Timeout waiting for Ollama server")
    return False

class Predictor(BasePredictor):
    def setup(self):
        """Setup necessary resources for predictions"""
        # set environment variable OLLAMA_MODELS to 'checkpoints'
        os.environ["OLLAMA_MODELS"] = MODEL_CACHE

        # Download weights - comment out to use ollama to donwload the weights
        print("Downloading weights")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Start server
        print("Starting ollama server")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the server to start
        if not wait_for_ollama():
            raise RuntimeError("Failed to start Ollama server")

        # Load model
        print("Running model")
        subprocess.check_call(["ollama", "run", MODEL_NAME], close_fds=False)

    def predict(self,
        image: Path = Input(description="Input image for the model"),
        prompt: str = Input(description="Input text for the model"),
        temperature: float = Input(description="Controls randomness. Lower values make the model more deterministic, higher values make it more random.", default=0.7, ge=0.0, le=1.0),
        top_p: float = Input(description="Controls diversity of the output. Lower values make the output more focused, higher values make it more diverse.", default=0.95, ge=0.0, le=1.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate", default=512, ge=1),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model and stream the output"""
        # Encode image
        encoded_image = encode_image(str(image))

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image]
                }
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        with requests.post(
            OLLAMA_CHAT,
            headers=headers,
            data=json.dumps(payload),
            stream=True,
            timeout=60
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'message' in chunk and 'content' in chunk['message']:
                            yield chunk['message']['content']
                    except json.JSONDecodeError:
                        print("Failed to parse response chunk as JSON")
