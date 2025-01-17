import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        Load csv data and return equivalent pandas DataFrames
    input:
        - messages_filepath : string, path to messages dataset
        - categories_filepath : string, path to categories dataset
    output:
        - df : pandas dataframe, merged messages and categories dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on=['id'])
    return df


def clean_data(df):
    '''
        clean and organize dataframe to be used for future training
    input:
        - df : pandas dataframe, data to be cleaned
    output:
        - df : pandas dataframe, cleaned dataframe
    '''
    categories_df = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories_df.iloc[0]
    # remove last 2 characters
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories
    categories_df.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories_df[column] = pd.to_numeric(categories_df[column])

    
    df = df.drop(columns=['categories'])
    df = pd.concat([df,categories_df], axis=1)
    df['related'] = df['related'].replace([2.0], df['related'].mode())
    
    df = df.drop_duplicates(subset=['id'], keep='first')
    df = df.drop_duplicates(subset=['message'], keep='first')

    return df


def save_data(df, database_filename):
    '''
        Save df to sql database file
    input:
        - df : pandas dataframe, dataframe to be saved
        - database_filename : string, path name to be used for saving dataframe
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


def main():
    '''
        Main function that executes the program
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()