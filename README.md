# Disaster Response Pipeline Project

### Project Summary:
In this project, I tackle the problem of classifying disaster messages to allow for a fast and accurate response
by the respective authorities. A full pipeline is built where I use the messages dataset from [Figure Eight](https://appen.com/datasets/combined-disaster-response-data/), clean
the data for training, train a machine learning model on the cleaned data and deploy the model on a flask-based webapp.

### Description of files:

The diagram below shows the repo directory tree and description for each file:
-app
	->templates
    	->go.html - shows the classification results of the given messages on the webapp
        ->master.html - the main html and javascript code for the webapp
    ->run.py - script responsible for running the webapp
-data
	->DisasterResponse.db - Database created after cleaning the dataset
    ->disaster_categories.csv - corresponding categories for each message
    ->disaster_messages.csv - disaster response messages
    ->process_data.py - ETL pipeline
-model
	->classifier.pkl - machine learning model saved to disk
    ->train_classifier.py - Machine learning pipeline

### python libraries needed:

Code was developed in Python 3.6.3 and the following libraries are needed:
- sklearn
- nltk
- re
- pickle
- pandas
- numpy
- sqlalchemy
- plotly
- Flask

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements

Data used was obtained from Udacity in collaboration with [Figure Eight](https://appen.com/datasets/combined-disaster-response-data/)

Otherwise, feel free to use the code!
