# Disaster Response Text Classification
A web app to help emergency response to disaster. Based on ML methods, it classify the disaster data from [Appen](https://appen.com/) into different event categories, and send the messages to an appropriate disaster relief agency.


## Dependencies
Install the python libraries defined in `requirements.txt` 
```
pip install -r requirements.txt
```

## Run instructions:

Run the following commands in the project's **root directory** to set up the database, train and save model, finally run the web app


Step 1: to run ETL pipeline that cleans data and stores in database
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

Step 2: to run ML pipeline that trains classifier and saves. If you want to try different ML algorithms and tune the model to get higher accuracy, then add `--tunning` as agurunment to following command line.
```
python models/train_classifier.py --database_filepath data/DisasterResponse.db --model_filepath models/classifier.pkl
```


Step 3: to run the web app  
```
python app/run.py
```

Step 4: go to http://127.0.0.1:3001/


## Files
<pre>
.
├── app
│   ├── run.py
│   └── templates
│       ├──go.html
│       └── master.html
├── data
│ ├── DisasterResponse.db
│ ├── disaster_categories.csv
│ ├── disaster_messages.csv
│ └── process_data.py
├── models
│ └── train_classifier.py
├── .gitignore
├── README.md
├── poetry.lock
└── pyproject.toml
 
</pre>