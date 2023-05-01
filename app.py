from flask import Flask
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import datetime
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/train')
def get_data_and_train():
    data_path='./result/model_with_data'
    data = pd.read_csv('house_price.csv')
    data=data[:60].copy()
    data.to_csv('./result/model_with_data/training_data.csv')
    X = data[['area', 'bedrooms', 'bathrooms', 'stories']]
    y = data['price']   
    model = LinearRegression()
    rows=len(X)
    model.fit(X, y)
    joblib.dump(model, './result/model_with_data/house_price_prediction_model.pkl')
    now = datetime.datetime.now()
    filename = 'result/model_with_data/model_details.txt'
    with open(filename, 'w') as f:
        f.write(str(f'model trained at {now} with {rows} rows of data') + '\n')
    # os.system(f'dvc add {data_path}')
    # os.system('dvc push')
    # os.system('git add .')
    # os.system(f"git commit -m 'updated training data to have {rows} rows and trained new model at {now}'")
    # os.system('git push origin main')
    # Add the DVC-tracked files to the DVC cache and push them to your remote storage
    subprocess.run(['dvc', 'add', './result/model_with_data'])
    subprocess.run(['dvc', 'push'])
    # Generate a commit message with the current timestamp
    commit_message = f'Trained model at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with {rows} of data'
    # Commit the changes to Git with the generated commit message
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', commit_message])
    subprocess.run(['git', 'push'])
    
    
    
    return 'Model trained and changes committed to DVC and Git!'



if __name__ == '__main__':
    app.run(debug=True)