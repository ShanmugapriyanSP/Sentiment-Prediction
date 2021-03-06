from flask import render_template, request, flash, url_for
import re
import nltk
from werkzeug.utils import redirect
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sentiment_prediction import app
import pickle

def predict_review_type(review_test):
    sparser = pickle.load(open('sentiment_prediction/files/sparser.pkl', 'rb'))
    model = pickle.load(open('sentiment_prediction/files/model.pkl', 'rb'))
    scaler = pickle.load(open('sentiment_prediction/files/scaler.pkl', 'rb'))

    review_test = re.sub('[^a-zA-Z]', ' ', review_test)
    review_test = review_test.lower()
    review_test = review_test.split()
    ps = PorterStemmer()
    review_test = [ps.stem(word) for word in review_test if not word in set(stopwords.words('english'))]
    review_test = ' '.join(review_test)
    review_test = [review_test]
    review_test = sparser.transform(review_test).toarray()
    review_test = scaler.transform(review_test)
    result = "Positive Review" if model.predict(review_test)[0] == 1 else "Negative Review"
    return result


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', title='ML - Sentiment Analysis')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        test_review = request.form['review']
        if test_review == '':
            result = ''
        else:
            result = predict_review_type(test_review)
    except:
        print('Exception')
        return redirect(url_for('home'))
    return render_template('home.html', result=result, title='Prediction')
