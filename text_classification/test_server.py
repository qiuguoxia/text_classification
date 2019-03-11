# /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import json
from flask import Flask, jsonify, request
import re

import xgboost as xgb
import pandas as pd
from xgboost_test_classification import readtrain
from xgboost_test_classification import segmentWord
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost_CHI_text_classification import process_file
from xgboost_CHI_text_classification import calculate_B_from_A
from xgboost_CHI_text_classification import feature_select_use_new_CHI
from xgboost_CHI_text_classification import document_features


app = Flask(__name__)

#根据列表中是否为空，将不为空的配成键值对更新到字典中



@app.route('/nlp/text_classification', methods=['GET', 'POST'])
def text_classification():

    if request.method == 'GET':
        text = request.args.get('input')
    if request.method == 'POST':
        text = request.form.get('input')
    content = re.sub(r'[^\w\s]', '', text)
    print("-------->")
    #content ='1'+','+content
    #content = "打开客厅的灯"
    with open('data/test_input.csv', 'w', encoding='utf8') as f:
        f.write('1')
        f.write(',')
        f.write(content)
        f.write('\n')
    A, tf, tf2, train_set, test_set, count, train_label = process_file('data/training.csv', 'data/test_input.csv')
    word_features = feature_select_use_new_CHI(A, calculate_B_from_A(A), count)
    test_documents_feature = [document_features(word_features, tf2, data[0], i)
                              for i, data in enumerate(test_set)]
    bst2 = xgb.Booster(model_file='xgb.model')
    dtest2 = xgb.DMatrix(test_documents_feature)
    preds2 = bst2.predict(dtest2)
    categoryNumber = int(preds2)+1
    if categoryNumber == 1:
          domain = "musicX"
    if categoryNumber == 2:
          domain = "story"
    if categoryNumber == 3:
           domain ="smartHome"
    if categoryNumber == 4:
           domain ="other"
    #result = result_to_json(content,categoryNumber)
    #str = content+":"+domain
    str = {"Query":content,"domain":domain}

    j = json.dumps(str)
    return j



if __name__ == '__main__' :
    app.run(host='172.28.80.204', port=int("15000"),debug=True)
    #app.run(host='localhost', port=int("5051"),debug=True)
