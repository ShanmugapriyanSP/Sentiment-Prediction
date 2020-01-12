# Importing the libraries
import pickle
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestClassifier

# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results  
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

model.fit(X_train, y_train)
# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

tn = cm[0, 0]
tp = cm[1, 1]
fn = cm[0, 1]
fp = cm[1, 0]

accuracy = (tp + tn) / (tp + tn + fn + fp)
model.fit(X, y)
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('sparser.pkl', 'wb'))


def predictResponse(review_test):
    review_test = re.sub('[^a-zA-Z]', ' ', review_test)
    review_test = review_test.lower()
    review_test = review_test.split()
    ps = PorterStemmer()
    review_test = [ps.stem(word) for word in review_test if not word in set(stopwords.words('english'))]
    review_test = ' '.join(review_test)
    review_test = [review_test]
    sparser = pickle.load(open('sparser.pkl', 'rb'))
    review_test = sparser.transform(review_test).toarray()
    model_file = pickle.load(open('model.pkl', 'rb'))
    return "Positive" if model_file.predict(review_test)[0] == 1 else "Negative"


print(predictResponse('Good'))
print(predictResponse('Crust is not good.'))