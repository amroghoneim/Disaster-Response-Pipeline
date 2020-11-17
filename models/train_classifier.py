# import libraries
import re
import sys
import nltk
import warnings
warnings.filterwarnings('ignore')
nltk.download(['punkt', 'wordnet', 'stopwords'])
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report, f1_score


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(engine.table_names()[0], con = engine)
    # remove unneccessary columns for training
    df = df.drop(columns=['id','original','genre'])
    # remove null rows
    df = df.dropna()
    X = df['message']
    y = df.drop(columns=['message'])
    category_names = df.drop(columns=['message']).columns.values
    return X, y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('tfidf-vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', 
                                                             n_jobs=10, 
                                                             max_leaf_nodes=4,
                                                             n_estimators=10,
                                                             random_state=42)))])
         
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)
    start = time.time()
    print(classification_report(Y_test, y_pred, target_names=category_names))    
    end = time.time()
    print('testing time: {}'.format(end-start))


def save_model(model, model_filepath):
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        parameters = {
             'tfidf-vect__ngram_range': ((1, 1), (1, 2))
        }

        cv = GridSearchCV(model, param_grid=parameters, verbose=1, n_jobs=-1, scoring='f1_weighted')
        cv.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()