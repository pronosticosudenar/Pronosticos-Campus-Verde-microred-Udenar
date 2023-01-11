import streamlit as st 
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
from sklearn.preprocessing import MinMaxScaler
from numpy.matrixlib.defmatrix import concatenate

import keras
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import learning_rate_schedule





dataset=pd.read_csv('https://raw.githubusercontent.com/sadoky/Pandas/master/1.232462_-77.293538_Solcast_PT5M.csv')
set_entrenamiento1= (dataset["Ghi"].iloc[0:4000])
set_validacion1 = dataset["Ghi"].iloc[4000:6240]


sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento=set_entrenamiento1.values.reshape(-1,1)
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)


time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50


modelo = Sequential()
modelo.add(GRU(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer="adam", loss='mse')
history=modelo.fit(X_train,Y_train,epochs=20,batch_size=32,verbose=0)


x_test = set_validacion1.values
x_test=x_test.reshape(-1,1)
x_test = sc.transform(x_test)


    
X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
    

prediccion = modelo.predict(x_test)
prediccion = sc.inverse_transform(prediccion)
prediccion2=pd.DataFrame(prediccion)

xaxis=np.arange(start=0,stop=43500,step=5)



url='https://raw.githubusercontent.com/sadoky/Pandas/master/1.232462_-77.293538_Solcast_PT5M.csv'
df=pd.read_csv(url)
datos_train=df["Ghi"].iloc[0:4000]
datos_val=df["Ghi"].iloc[4001:6240]


j=np.zeros(4000,int)
pred44=pd.DataFrame(j)
pred56=np.asarray(prediccion2,dtype=np.float64)

pred56=pred56.reshape(-1,1)
pred57=j.reshape(-1,1)

lim=np.concatenate((pred57,pred56))
lim2=pd.DataFrame(lim)
prediccion_final=lim2[4000:]





header = st.container()
dataset=st.container()
features= st.container()
modelTraining= st.container()

with header:
    st.title("Prediccion de irradiancia campus verde")
    
with dataset:
    st.header("Set de datos estacion campus verde ")



    taxi_data=pd.read_csv(r"C:\Users\santiago\Documents\APY_VS_CODE\APY_CONTENT\data\cream.csv")
    st.write(taxi_data.head())

    datos_train=taxi_data["Ghi"].iloc[0:4000]
    st.line_chart(datos_train)
    
    st.header("propiedades de los datos y estadisticas ")

    

    st.text("Cantidad de datos duplicados: 4398 ")
    st.text("Cantidad de datos nulos: 0  ")

    st.subheader("Propiedades estadísticas ")

    st.metric("count",'161218.000000')
    st.metric("mean " ,'169.965496')
    st.metric("std"," 264.015458")
    st.metric('min ' ,'0')
    st.metric(' 25%  ','2.2')
    st.metric('50% ',' 6.9')
    st.metric(' 75% ',' 267.5')
    st.metric('max ',' 1514.9')


    st.subheader('Metricas')

    st.metric('MSE','6848.6455078125')
    st.metric('Testing RMSE','827565')
    st.metric('MAE',' 38.99604415893555')
    st.metric('R2','0.8839807918877545')
    
    







with features:
    st.subheader('Prediccion final')
    st.line_chart(prediccion_final)

    


    

