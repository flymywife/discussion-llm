import cohere
import gradio as gr
from dotenv import load_dotenv
import os

# .envファイルから環境変数を読み込む
load_dotenv()

co = cohere.Client(os.getenv('COHERE_API_KEY'))

def chat_model1_to_model2(message, chat_history=[]):
    response1 = co.chat(model="command-r", message=message, chat_history=chat_history,preamble="Please discuss the global environment")
    response_text1 = response1.text
    chat_history.append({"role": "USER", "text": message})
    chat_history.append({"role": "CHATBOT", "text": response_text1})
    
    response2 = co.chat(model="command-r-plus", message=response_text1, chat_history=chat_history,preamble="Please discuss the global environment")
    response_text2 = response2.text
    chat_history.append({"role": "USER", "text": response_text1})
    chat_history.append({"role": "CHATBOT", "text": response_text2})
    
    return response_text1, response_text2, chat_history

def gradio_interface(message):
    chat_history = []
    response_text1, response_text2, chat_history = chat_model1_to_model2(message, chat_history)
    return f"Model 1: {response_text1}\n\nModel 2: {response_text2}"

gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="Model-to-Model Discussion").launch()
