import flask as fl
from flask import jsonify, request
import pickle
import numpy as np

app=fl.Flask(__name__)

#load index file
@app.route('/')
def indext():
    return app.send_static_file('index.html')

    
@app.route('/api/linear/<int:inp>')
def linear(inp):
    Xpred=[[inp]]
    load_reg=pickle.load(open('LR.pkl', 'rb'))
    result=load_reg.predict(Xpred)
    res=str(result[0])
    return res
   
@app.route('/api/randomforest/<int:inp>')
def randomforest(inp):
    Xpred=[[inp]]
    load_RF=pickle.load(open('RF.pkl', 'rb'))
    result=load_RF.predict(Xpred)
    res=str(result[0])
    return res  
    


