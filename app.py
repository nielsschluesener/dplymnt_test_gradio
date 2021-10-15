import gradio as gr
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

def make_prediction(sepal_length, sepal_width, petal_length, petal_width):
    pred_array = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    pred = model.predict(pred_array)
    if pred == 0:
        flower = "Iris Setosa"
    elif pred == 1:
        flower ="Iris Versicolour"
    elif pred == 2:
        flower = "Iris Virginica"
    return flower

iface = gr.Interface(
    fn=make_prediction, 
    inputs = ["number","number","number","number"],
    outputs = "text")

iface.launch()