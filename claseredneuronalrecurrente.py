# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:55:02 2021

@author: Alan Covarrubias
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import random
import pickle
import time
from PIL import Image

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

def remove(string):
    a=string.replace(':','-')
    return a.replace(" ", "")    

class RNN:
    
    def __init__(self,salida,tamaño,NOMBRE=remove(time.asctime()[0:11])+remove(time.asctime()[13:19])):
        
        

        #opt=tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

        
        self.modelo=tf.keras.models.Sequential([tf.keras.layers.LSTM(128,input_shape=(140,140), return_sequences=True),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(32, activation='relu'),
                                    tf.keras.layers.LSTM(128,activation='relu'),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(32, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(salida,activation='softmax')
                                    ])
    
        
        self.modelo.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])
        
        
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NOMBRE))
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint("models/{}.model".format(NOMBRE, monitor='val_accuracy', 
                                                                                      verbose=1, save_best_only=True, mode='max'))
        self.modelo.save('models/{}'.format('Modelo Recurrente'))
        
        self.modelo.summary()
        
    def historia(self,datos_entrenamiento,epochs,datos_validacion):
        
        print('tenemos %d imagenes de %d pixeles' %(len(datos_entrenamiento),datos_entrenamiento.shape[1]))
        time.sleep(2)
        
        self.historia=self.modelo.fit(datos_entrenamiento,
                                      datos_validacion,
                                      batch_size=16,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_split=0.3,
                                      callbacks=[self.tensorboard])#,self.checkpoint])
        
        
        
        #prueba_loss, prueba_acc = self.modelo.evaluate(datos_validacion, verbose=2)


    def graficos(self, historia,epocas):
        
        x=[i for i in range(epocas)]
       
        plt.figure(0)
        plt.scatter(x,self.historia.history['accuracy'],c='blue')
        plt.plot(x,self.historia.history['accuracy'],label='accuracy',color='blue')
        
        plt.ylabel('accuracy')
        plt.xlabel('epocas')
        plt.legend(loc='lower right')
        
        plt.ylabel('loss')
        plt.xlabel('epocas')
        plt.scatter(x,self.historia.history['loss'],c='r')
        plt.plot(x,self.historia.history['loss'],color='red',label='loss')
        plt.ylim(0,1.2)
        plt.legend(loc='lower right')
        plt.rcParams.update({'axes.facecolor':(1.0,0.93,0.8039)})
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        plt.show()
        
        plt.figure(1)
        
        plt.scatter(x,self.historia.history['accuracy'],c='blue')
        plt.plot(x,self.historia.history['accuracy'],label='accuracy',color='blue')
        
        plt.ylabel('accuracy')
        plt.xlabel('epocas')
        plt.legend(loc='lower right')
        
        plt.ylabel('val_accuracy')
        plt.xlabel('epocas')
        plt.scatter(x,self.historia.history['val_accuracy'],c='r')
        plt.plot(x,self.historia.history['val_accuracy'],color='red',label='val_accuracy')
        plt.ylim(0,1.2)
        plt.legend(loc='lower right')
        plt.rcParams.update({'axes.facecolor':(1.0,0.93,0.8039)})
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        plt.show()
        
    
    def predicciones(self,ruta,img_size,datadir):
        categorias=[i for i in os.listdir(datadir)]
        prediccion=[]
        img_array=cv2.imread(ruta, cv2.IMREAD_GRAYSCALE) 
        new_array=cv2.resize(img_array, (img_size,img_size))
        prediccion.append([new_array])
        prediccion=np.array(prediccion).reshape(img_size,img_size, 3)
        prediccion = prediccion.astype('float32')
        prediccion /= 255 
        a=self.modelo.predict(prediccion)
        a[0]=list(a[0])
        jaj=np.argmax(a[0], axis=0)
        print('Hay %2.2f de probabilidad de que sea %s' %(max(a[0]),categorias[jaj]))
        print(a)
        
        
def creador_datos_entrenamiento(datadir,img_size):
    a=img_size
    categorias=[i for i in os.listdir(datadir)]
    train_data = []
    X= []  #Conjunto de caracteristicas
    y= []
    
    for cat in categorias:
        path = os.path.join(datadir,cat) #/placas/categoria/
        clase_num = categorias.index(cat)
        for img in os.listdir(path): #/placas/categoria/imagen
            try:
                img_array=cv2.imread(os.path.join(path,img))#, cv2.IMREAD_GRAYSCALE) 
            #reducimos a una escala de grises
                new_array=cv2.resize(img_array, (a,a))
                train_data.append([new_array,clase_num])
            except Exception as e:
                pass
    random.shuffle(train_data)
    
    try:
        os.remove('X.pickle')
        os.remove('y.pickle')
    except Exception as e:
        pass

    for caracteristicas,etiqueta in train_data:
        X.append(caracteristicas)
        y.append(etiqueta)
    X=np.array(X).reshape(-1,img_size,img_size, 3) #El uno hace referencia al canal de color
    
    pickle_out=open('X.pickle', 'wb')
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out=open('y.pickle', 'wb')
    pickle.dump(y,pickle_out)
    pickle_out.close()
    return categorias