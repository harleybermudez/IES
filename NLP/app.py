import os
import nltk
import spacy
from flask import Flask, render_template, request
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# --- Download required NLTK data (only once) ---
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Load spaCy model (small English model) ---
# If you haven't installed it yet, run: python -m spacy download en_core_web_sm
nlp_spacy = spacy.load("en_core_web_sm")

# Get the directory of the current script
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))
app = Flask(__name__)

# Initialize tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html", result=None, input_text="")

@app.route("/process", methods=["POST"])
def process():
    """Process the user's text based on selected task."""
    input_text = request.form.get("input_text", "")
    task = request.form.get("task", "tokenize")
    result = ""

    if not input_text.strip():
        result = "Please enter some text."
    else:
        if task == "tokenize":
            # Tokenization using NLTK
            tokens = word_tokenize(input_text)
            result = f"Tokens: {tokens}"

        elif task == "pos_tag":
            # Part-of-Speech tagging using TextBlob (or NLTK)
            blob = TextBlob(input_text)
            pos_tags = blob.tags  # list of (word, tag) tuples
            result = "POS Tags:\n" + "\n".join([f"{word} -> {tag}" for word, tag in pos_tags])

        elif task == "ner":
            # Named Entity Recognition using spaCy
            doc = nlp_spacy(input_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                result = "Named Entities:\n" + "\n".join([f"{text} -> {label}" for text, label in entities])
            else:
                result = "No named entities found."

        elif task == "sentiment":
            # Sentiment analysis using TextBlob
            blob = TextBlob(input_text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            if polarity > 0:
                sentiment = "Positive 😊"
            elif polarity < 0:
                sentiment = "Negative ☹️"
            else:
                sentiment = "Neutral 😐"
            result = (f"Sentiment: {sentiment}\n"
                      f"Polarity score: {polarity:.2f}\n"
                      f"Subjectivity score: {subjectivity:.2f}")

        elif task == "stemming":
            # Stemming using NLTK PorterStemmer
            tokens = word_tokenize(input_text)
            stems = [stemmer.stem(token) for token in tokens]
            result = f"Original tokens: {tokens}\nStems: {stems}"

        elif task == "lemmatization":
            # Lemmatization using NLTK WordNetLemmatizer
            tokens = word_tokenize(input_text)
            lemmas = [lemmatizer.lemmatize(token) for token in tokens]
            result = f"Original tokens: {tokens}\nLemmas: {lemmas}"

        else:
            result = "Unknown task selected."

    return render_template("index.html", result=result, input_text=input_text, selected_task=task)

if __name__ == "__main__":

    app.run(debug=True)
