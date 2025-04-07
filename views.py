from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize NLP Model
lemmatizer = WordNetLemmatizer()

# Debugging: File loading
try:
    print("Loading intents.json...")
    intents = json.loads(open('chatbot/intents.json').read())
    print("✅ intents.json loaded successfully!")

    print("Loading words.pkl...")
    words = pickle.load(open('chatbot/words.pkl', 'rb'))
    print("✅ words.pkl loaded successfully!")

    print("Loading classes.pkl...")
    classes = pickle.load(open('chatbot/classes.pkl', 'rb'))
    print("✅ classes.pkl loaded successfully!")

    print("Loading chatbot model...")
    model = load_model('chatbot/chatbotmodel.h5')
    print("✅ Chatbot model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading files: {e}")

# Preprocessing Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results] if results else []

# Chatbot API Route
@csrf_exempt  # ✅ CSRF fix
def get_response(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()

        if not user_message:
            return JsonResponse({"response": "Sorry, I didn't understand."})

        ints = predict_class(user_message)
        if not ints:
            return JsonResponse({"response": "I'm not sure how to respond to that."})

        response = random.choice([resp for intent in intents["intents"] if intent["tag"] == ints[0]["intent"] for resp in intent["responses"]])

        return JsonResponse({"response": response})

    except Exception as e:
        print(f"❌ Error in chatbot response: {e}")
        return JsonResponse({"error": "Something went wrong."}, status=500)

# Render Chatbot Page
def index(request):
    return render(request, "chatbot/index.html")
