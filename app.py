from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np # Import numpy

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    print("Please run the 'train_model.ipynb' notebook first to create them.")
    exit() # Exit if files are not found

# Download stopwords (needed for the cleaning function)
nltk.download('stopwords')

# --- Re-create the SAME cleaning function from the notebook ---
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    cleaned_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)
# -----------------------------------------------------------------

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Get the tweet from the form
        tweet_text = request.form['tweet']
        
        # 2. Clean the tweet
        cleaned_text = clean_tweet(tweet_text)
        
        # 3. Vectorize the tweet
        #    Note: We use vectorizer.transform() here, NOT fit_transform()
        text_vector = vectorizer.transform([cleaned_text])
        
        # 4. Make prediction (get probabilities)
        #    model.predict_proba() returns [[prob_neg, prob_pos]]
        probabilities = model.predict_proba(text_vector)[0]
        
        # 5. Get the confidence score
        confidence = round(max(probabilities) * 100, 2)
        
        # 6. Get the prediction label
        prediction_code = np.argmax(probabilities) # 0 or 1
        sentiment = 'Positive' if prediction_code == 1 else 'Negative'

        # 7. Get individual probabilities for the chart
        prob_neg = round(probabilities[0] * 100, 2)
        prob_pos = round(probabilities[1] * 100, 2)
                
        # 8. Send all data to the result.html template
        return render_template(
            'result.html', 
            prediction=sentiment, 
            tweet=tweet_text,
            confidence=confidence,
            prob_neg=prob_neg,
            prob_pos=prob_pos
        )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)