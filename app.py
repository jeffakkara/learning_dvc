from flask import Flask
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/train')
def get_data_and_train():
    data = pd.read_csv('house_price.csv')
    data=data[:10].copy()
    data.to_csv('./result/model_with_data/training_data.csv')
    X = data[['area', 'bedrooms', 'bathrooms', 'stories']]
    y = data['price']   
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, './result/model_with_data/house_price_prediction_model.pkl')

    return 'Done'




if __name__ == '__main__':
    app.run(debug=True)