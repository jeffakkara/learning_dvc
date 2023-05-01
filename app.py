from flask import Flask
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import datetime
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/train')
def get_data_and_train():
    data_path='./result/model_with_data'
    data = pd.read_csv('house_price.csv')
    data=data[:25].copy()
    data.to_csv('./result/model_with_data/training_data.csv')
    X = data[['area', 'bedrooms', 'bathrooms', 'stories']]
    y = data['price']   
    model = LinearRegression()
    rows=len(X)
    model.fit(X, y)
    joblib.dump(model, './result/model_with_data/house_price_prediction_model.pkl')
    now = datetime.datetime.now()
    filename = 'result/model_with_data/model_details.txt'
    with open(filename, 'a') as f:
        f.write(f'model trained at {now} with {rows} rows of data' + '\n')
    os.system(f'dvc add {data_path}')
    os.system('dvc push')
    os.system('git add .')
    os.system('git commit -m f"updated training data to have {rows} rows and trained new model at {now}"')
    os.system('git push origin main')
    return 'Done'




if __name__ == '__main__':
    app.run(debug=True)