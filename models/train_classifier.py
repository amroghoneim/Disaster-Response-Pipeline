# import libraries
import re
import sys
import nltk
import warnings
warnings.filterwarnings('ignore')
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer



def load_data(database_filepath):
    '''
        Load data from database into pandas dataframe and split into features, labels and category names
    input:
        - database_filepath : pandas dataframe, data to be used for training
    output:
        - X : Pandas Series, messages data to be used for training
        - y : Pandas Series, messages corresponding multi-label. binary values (0-1)
        - category_names : List, Name for each label
    '''
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
    '''
        Function to transform text data into clean tokens
    input:
        - text : List, text data to be converted into tokens
    output:
        - clean_tokens : List, cleaned tokens
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    # remove stop words
    stop_words = stopwords.words('english')
    tokens = [w for w in tokens if w not in stop_words]
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''
        Function to build machine learning pipeline
    input:
        - None
    output:
        - pipeline : Sklearn Pipeline Object, machine learning pipeline
    '''
    pipeline = Pipeline([
        ('tfidf-vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', 
                                                             n_jobs=10, 
                                                             max_leaf_nodes=4,
                                                             n_estimators=10,
                                                             random_state=42)))])
    parameters = {
             'tfidf-vect__ngram_range': ((1, 1), (1, 2))
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1, scoring='f1_weighted')
         
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Function to evaluate machine learning model performance
    input:
        - model: Sklearn Pipeline Object, machine learning model to be used for prediction
        - X_test: List, test data
        - Y_test: List, test data labels
        - category_names: List, label names
    output:
        - None
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))    


def save_model(model, model_filepath):
    '''
        Function to save machine learning model to disk
    input:
        - model : Sklearn Pipeline Object, model to be saved
        - model_file_path : string, path used to save model
    output:
        - None
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
        Main function that executes the program
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')

        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()