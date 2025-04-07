import random
import json
import pickle
import numpy as np
import nltk
import os
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD

# NLTK Downloads (Important for Tokenizer)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load Intents File
file_path = os.path.join(os.getcwd(), 'chatbot_project', 'chatbot', 'intents.json')
  # Current directory se read karega
  # Safer file path
with open(file_path, 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Data Preprocessing
words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize sentence
        words.extend(word_list)
        documents.append((word_list, intent['tag']))  # Store word list with tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize & Clean Words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Remove duplicates
classes = sorted(set(classes))  # Sort class labels

# Save Processed Data
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Prepare Training Data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    
    output_row = output_empty[:]
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle & Convert to NumPy Array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([np.array(i, dtype=np.float32) for i in training[:, 0]])
train_y = np.array([np.array(i, dtype=np.float32) for i in training[:, 1]])

# Define Model
model = Sequential([
    Input(shape=(len(train_x[0]),)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile Model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train Model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save Model
model.save("chatbotmodel.h5")

print("Model Training Completed & Saved Successfully!")
