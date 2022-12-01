import dill
from datetime import datetime
import pandas as pd
import os
import json




def predict():
    path = '../data/test/'
    file_name = '../data/models/cars_pipe_202211292038.pkl'

    with open(file_name, 'rb') as file:
        model = dill.load(file)

    result = pd.DataFrame()

    for filename in os.listdir(path):
        data = pd.DataFrame([pd.read_json(path + filename, typ='series')])
        y = model.predict(data)
        result = result.append([[filename[:-5], y[0]]])

    result.columns = ['file number', 'predict']
    result.to_csv('../data/predictions/predict1.csv', index=False)


if __name__ == '__main__':
    predict()

