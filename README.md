# Twitter Sentiment Analysis 🐦

A machine learning-powered web application that analyzes the sentiment of tweets and text content. Built with Flask and scikit-learn, this project uses natural language processing to classify text as positive or negative with confidence scores.

## Features

- **Real-time Sentiment Analysis**: Instantly analyze any text or tweet
- **Confidence Scoring**: Get percentage-based confidence levels for predictions
- **Visual Analytics**: Interactive doughnut chart showing sentiment probabilities
- **Clean UI**: Modern, Twitter-inspired interface
- **Text Preprocessing**: Advanced cleaning with stopword removal and stemming

## Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, NLTK
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Chart.js
- **NLP Tools**: PorterStemmer, TF-IDF Vectorization

## Project Structure

```
twitter-sentiment-analysis/
│
├── app.py                      # Flask application and routes
├── train_model.ipynb           # Model training notebook (required)
├── model.pkl                   # Trained ML model (generated)
├── vectorizer.pkl              # TF-IDF vectorizer (generated)
│
├── templates/
│   ├── index.html             # Home page
│   └── result.html            # Results display page
│
└── static/
    └── style.css              # Styling
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager
- Dataset Link:https://www.kaggle.com/datasets/kazanova/sentiment140

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd twitter-sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install flask nltk scikit-learn numpy
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Train the model**
   - Run the `train_model.ipynb` notebook to generate `model.pkl` and `vectorizer.pkl`
   - These files must exist before running the Flask app

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the app**
   - Open your browser and navigate to `http://127.0.0.1:5000`

## How It Works

### Text Preprocessing Pipeline

1. **URL Removal**: Strips out all HTTP/HTTPS links
2. **Mention Removal**: Removes @username mentions
3. **Hashtag Removal**: Cleans hashtag symbols
4. **Special Character Removal**: Keeps only alphabetic characters
5. **Lowercase Conversion**: Normalizes text case
6. **Stopword Removal**: Filters common English words
7. **Stemming**: Reduces words to root form using Porter Stemmer

### Prediction Process

1. User submits text via the web form
2. Text is cleaned using the preprocessing pipeline
3. Cleaned text is vectorized using TF-IDF
4. ML model predicts sentiment probabilities
5. Results displayed with confidence score and visualization

## Usage Example

**Input:**
```
"I absolutely love this new feature! It's amazing and works perfectly!"
```

**Output:**
- Sentiment: Positive
- Confidence: 95.6%
- Visual chart showing probability distribution

## Model Details

- **Algorithm**: (Specify your model - e.g., Logistic Regression, Naive Bayes, etc.)
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Training Data**: (Add details about your dataset)
- **Performance Metrics**: (Add accuracy, precision, recall if available)

## API Routes

- `GET /` - Home page with input form
- `POST /predict` - Sentiment prediction endpoint

## Future Enhancements

- [ ] Add support for neutral sentiment
- [ ] Implement batch processing for multiple tweets
- [ ] Add emoji sentiment analysis
- [ ] Include historical analysis tracking
- [ ] Deploy to cloud platform (Heroku, AWS, etc.)
- [ ] Add API endpoint for external integrations
- [ ] Support for multiple languages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for natural language processing tools
- Chart.js for beautiful visualizations
- Flask framework for easy web development
- scikit-learn for machine learning capabilities

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: Make sure to train your model using `train_model.ipynb` before running the application. The app requires both `model.pkl` and `vectorizer.pkl` files to function.
