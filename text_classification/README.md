# text_classification

CHI选择特征词，TFIDF计算权重，朴素贝叶斯、XGBoost等算法，
类别标签
1 musicX
2 story
3 smartHome
4 other

train----
python xgboost_CHI_text_classification
test-----
python xgboost_test_classification 

起服务
请求格式 
http://172.28.80.204:15000/nlp/text_classification?input="我想听周杰伦的歌"
 

python test_server.py
