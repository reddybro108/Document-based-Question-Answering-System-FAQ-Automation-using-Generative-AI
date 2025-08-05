import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    words = word_tokenize(text)
    filtered = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
    return " ".join(filtered)

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])
