from flask import Flask, render_template, request, jsonify, session
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Load pre-trained model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"  # You can change this to other models
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Model loaded!")

@app.route('/')
def index():
    # Initialize or reset chat history
    session['chat_history_ids'] = None
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    
    # Tokenize user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Append to chat history or initialize if first message
    bot_input_ids = new_user_input_ids
    if 'chat_history_ids' in session and session['chat_history_ids'] is not None:
        # Convert session string back to tensor
        chat_history_ids = torch.tensor(session['chat_history_ids'])
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    
    # Generate a response with context from chat history
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature=0.8
    )
    
    # Store chat history in session
    session['chat_history_ids'] = chat_history_ids.tolist()
    
    # Decode and return response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)