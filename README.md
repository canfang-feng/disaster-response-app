# Disaster Response Pipeline Project

### Instructions:

Run the following commands in the project's root directory to set up the database, train and save model, finally run the web app

    - Step 1: to run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Step 2: to run ML pipeline that trains classifier and saves
        `python models/train_classifier.py --database_filepath data/DisasterResponse.db --model_filepath models/classifier.pkl`
    - Step 3: to run the web app  
        `python app/run.py`
    - Step 4: go to http://127.0.0.1:3001/


