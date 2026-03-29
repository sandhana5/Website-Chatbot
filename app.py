from flask import Flask, render_template, request, session
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from datetime import datetime
import re
import os
from spellchecker import SpellChecker  # Import pyspellchecker library

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'mysecret123'

# Absolute path to feedback.txt
FEEDBACK_FILE = r'C:\Users\sandhana\Documents\websiteproject\my_website\app_folder\feedback.txt'

# Initialize spell checker
spell = SpellChecker()

# Ensure feedback.txt exists at startup
if not os.path.exists(FEEDBACK_FILE):
    print(f"Creating feedback.txt at {FEEDBACK_FILE}")
    try:
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            f.write("")
        print("feedback.txt created successfully")
    except Exception as e:
        print(f"Failed to create feedback.txt: {e}")
else:
    print(f"feedback.txt already exists at {FEEDBACK_FILE}")

# Load dataset
def load_dataset():
    try:
        with open('../static/my_chatbot_dataset.json', 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {"intents": [{"tag": "default", "patterns": [], "keywords": [], "entities": [], "responses": ["Error loading data"]}]}

chatbot_data = load_dataset()

# Prepare training data
patterns = []
intent_tags = []
for intent in chatbot_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        intent_tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns)

# Function to correct spelling in user query
def correct_spelling(text):
    words = text.split()
    corrected_words = []
    for word in words:
        # Check if the word is misspelled and correct it
        if spell.correction(word) is not None and spell.correction(word) != word:
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text

def get_intent(user_message):
    # Correct spelling before processing
    corrected_message = correct_spelling(user_message)
    user_vector = vectorizer.transform([corrected_message.lower()])
    similarities = cosine_similarity(user_vector, pattern_vectors)[0]
    max_similarity = max(similarities)
    if max_similarity > 0.5:
        best_match_idx = similarities.argmax()
        return intent_tags[best_match_idx]
    return "default"

def store_feedback(query, corrected_query, response, matched=True):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    feedback_entry = (
        f"Timestamp: {timestamp}\n"
        f"Original Query: {query}\n"
        f"Corrected Query: {corrected_query}\n"
        f"Matched: {matched}\n"
        f"Response: {response}\n"
        f"{'-' * 50}\n"
    )
    try:
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(feedback_entry)
        print(f"Feedback written to {FEEDBACK_FILE}")
    except Exception as e:
        print(f"Error writing to feedback.txt: {e}")

def update_dataset_from_feedback():
    global chatbot_data, patterns, intent_tags, pattern_vectors
    try:
        with open('../static/my_chatbot_dataset.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset for update: {e}")
        return
    
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback.txt found for update")
        return
    
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            lines = f.read().split('-' * 50 + '\n')
            for entry in lines:
                if not entry.strip():
                    continue
                query = re.search(r'Original Query: (.*)', entry)
                corrected_query = re.search(r'Corrected Query: (.*)', entry)
                matched = re.search(r'Matched: (.*)', entry)
                response = re.search(r'Response: (.*)', entry)
                if query and corrected_query and matched and response:
                    query_text = query.group(1).strip()
                    corrected_text = corrected_query.group(1).strip()
                    matched_status = matched.group(1).strip() == "True"
                    response_text = response.group(1).strip()
                    
                    if matched_status:
                        continue
                    
                    matched_intent = None
                    for intent in data['intents']:
                        if any(keyword in corrected_text.lower() for keyword in intent['keywords']):
                            matched_intent = intent['tag']
                            break
                    
                    if matched_intent and matched_intent != 'default':
                        intent_dict = next(i for i in data['intents'] if i['tag'] == matched_intent)
                        if corrected_text not in intent_dict['patterns']:
                            intent_dict['patterns'].append(corrected_text)
                    else:
                        new_tag = f"custom_{len(data['intents'])}"
                        data['intents'].insert(-1, {
                            "tag": new_tag,
                            "patterns": [corrected_text],
                            "keywords": corrected_text.lower().split(),
                            "entities": [],
                            "responses": [response_text]
                        })
        
        with open('../static/my_chatbot_dataset.json', 'w') as f:
            json.dump(data, f, indent=4)
        
        chatbot_data = load_dataset()
        patterns.clear()
        intent_tags.clear()
        
        for intent in chatbot_data['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern.lower())
                intent_tags.append(intent['tag'])
        pattern_vectors = vectorizer.fit_transform(patterns)
        print("Dataset updated from feedback")
    except Exception as e:
        print(f"Error updating dataset: {e}")

def get_bot_response(user_message):
    corrected_message = correct_spelling(user_message)
    intent_tag = get_intent(user_message)  # Use original message to get intent, but corrected message is processed
    for intent in chatbot_data['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            response = re.sub(r'[^\x00-\x7F]+', '', response)
            store_feedback(user_message, corrected_message, response, matched=(intent_tag != "default"))
            return response
    response = random.choice(chatbot_data['intents'][-1]['responses'])
    response = re.sub(r'[^\x00-\x7F]+', '', response)
    store_feedback(user_message, corrected_message, response, matched=False)
    return response

@app.route('/')
def home():
    if 'messages' not in session:
        session['messages'] = []
    if 'chatbot_open' not in session:
        session['chatbot_open'] = False  # Default to closed
    return render_template('index.html', messages=session['messages'], chatbot_open=session['chatbot_open'])

@app.route('/chat', methods=['POST'])
def chat():
    if 'messages' not in session:
        session['messages'] = []
    if 'chatbot_open' not in session:
        session['chatbot_open'] = False
    
    user_message = request.form.get('message', '').strip()
    if not user_message:
        session['messages'].append({'type': 'bot', 'text': "Please type a message."})
    else:
        response = get_bot_response(user_message)
        session['messages'].append({'type': 'user', 'text': user_message})
        session['messages'].append({'type': 'bot', 'text': response})
    
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                unmatched_count = sum(1 for line in f if "Matched: False" in line)
            if unmatched_count >= 5:
                update_dataset_from_feedback()
                with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
                    f.write("")
                print("Feedback cleared after update")
    except Exception as e:
        print(f"Error checking feedback count: {e}")
    
    session['chatbot_open'] = session.get('chatbot_open', False)
    session.modified = True
    return render_template('index.html', messages=session['messages'], chatbot_open=session['chatbot_open'])

@app.route('/toggle_chatbot', methods=['POST'])
def toggle_chatbot():
    if 'chatbot_open' not in session:
        session['chatbot_open'] = False
    session['chatbot_open'] = not session['chatbot_open']  # Toggle the state
    session.modified = True
    return '', 204  # No content response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
  
