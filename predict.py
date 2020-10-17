import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import request
import plotly.express as px
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import metrics

data=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')

X=data[['Hours']]
Y=data['Scores']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0) 

model = LinearRegression()  
model.fit(X_train, y_train) 

y_prd = model.predict(X_test)

#hours = float(input("How many Hours you studied : "))
#hour=request.args.get(hours)
#user_input = st.number_input("Hours you studied : ", 5254)
#st.write(user_input)
#print("No of Hours = {}".format(hours))
#print("Predicted Score = {:.2f} %".format(own_pred[0]))

import pickle
pickle_out=open("predict.pkl",'wb')
pickle.dump(model,pickle_out)
pickle_out.close()

pickle_in=open("predict.pkl",'rb')
predicter = pickle.load(pickle_in)

page_bg_img = '''
<style>
body {
background-image: url("https://cdn.pixabay.com/photo/2017/02/06/23/52/faded-2044616_1280.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title(" Predicting Score of Exam ")
st.markdown("### How many hour you study ?")
hour = st.slider("Hour of Day",0,10)
hr=np.reshape(hour,(-1,1))
st.markdown("### How much score you expect ?")
escore = st.slider("Hour of Day",0,100)
#own_pred = model.predict(hr)
result=''
if st.button("Predict Score"):
    result = model.predict(hr.astype(np.float64))
    result = float(result)
    if result >= 100:
        result=100
        st.success("Predicted Score is = {:.2f} % ".format(result))
    else:
        st.success("Predicted Score is = {:.2f} % ".format(result))

    #if st.checkbox("Show Plot",value=False):
        #dt=[escore,result]
        #df = pd.DataFrame(dt, columns = ['Score'])
        #st.write(df)
        #df.plot.bar(figsize=(10,6))
        #st.pyplot()
        dat = {'Expected Score ':escore,'Predicted Score':result}
        text = list(dat.keys())
        score = list(dat.values())
        plt.figure(figsize = (8,5))
        plt.bar(text,score,color='#666666',width=0.5)
        plt.ylabel("P e r c e n t a g e ") 
        plt.title("Expected Score v/s Predicted Score")
        plt.ylim(0,100)
        st.pyplot() 
