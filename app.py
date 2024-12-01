# app.py

from flask import Flask, render_template, request, jsonify, redirect, url_for
import sys
import os
from src.models.chat_with_improved_rick import ImprovedRickBot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    rickbot = ImprovedRickBot()
    logger.info("RickBot initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RickBot: {str(e)}")
    sys.exit(1)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_message = request.json['message'].strip().lower()
        
        # Handle special commands
        if user_message == 'quit':
            return jsonify({'command': 'quit'})
        elif user_message == 'clear':
            return jsonify({'command': 'clear'})
            
        # Handle normal messages
        if not user_message:
            return jsonify({'response': '*burp* Say something, Morty!'})
            
        # Handle science questions
        if user_message.startswith('science'):
            response = rickbot.get_scientific_response(user_message[7:].strip())
        else:
            response = rickbot.generate_response(user_message)
            
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({
            'response': "*burp* Something went wrong in dimension C-137!"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)