# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:02:36 2021

@author: Alan
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

class FNN:
    
    def __init__(self,salida,tamaño,NOMBRE=remove(time.asctime()[0:11])+remove(time.asctime()[13:19])):
        
        #salida=len([i for i in os.listdir(datadir)])
        
        self.modelo=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=tamaño),
                                                tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.experimental.preprocessing.Normalization(),
                                                tf.keras.layers.Dense(64,activation='sigmoid'),
                                                #tf.keras.layers.Dropout(0.3),
                                                tf.keras.layers.Dense(128,activation='relu'),
                                                tf.keras.layers.Dense(64,activation='sigmoid'),
                                                tf.keras.layers.Dropout(0.3),
                                                tf.keras.layers.Dense(256,activation='relu'),
                                                tf.keras.layers.Dense(salida,activation='softmax')])
        
        #SELECCIONAMOS EL OPTIMIZADOR
        
        self.modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['accuracy'])
        
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NOMBRE))
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint("models/{}.model".format(NOMBRE, monitor='val_accuracy',
                                                                                      verbose=1, save_best_only=True, mode='max'))
        #Guardamos el modelo 
        
        self.modelo.save('models/{}'.format('Modelotradicional'+NOMBRE))#)
        self.tamaño=np.array(tamaño)
        self.modelo.summary()
        
        
        
    def historia(self,datadir,epochs,minilote):
        
        
        
        tamaño_a=[i for i in self.tamaño[0:2]]
        tamaño_a=tuple(tamaño_a)
        
        datos_entrenamiento = tf.keras.preprocessing.image_dataset_from_directory(datadir,
                                                            labels='inferred',
                                                            label_mode='int',
                                                            class_names=[i for i in os.listdir(datadir)],
                                                            image_size=tamaño_a,
                                                            batch_size=minilote,
                                                            shuffle='True',
                                                            validation_split=0.2,
                                                            subset="training",
                                                            seed=123
                                                            )
        datos_validacion = tf.keras.preprocessing.image_dataset_from_directory(datadir,
                                                            labels='inferred',
                                                            label_mode='int',
                                                            class_names=[i for i in os.listdir(datadir)],
                                                            image_size=tamaño_a,
                                                            batch_size=minilote,
                                                            shuffle='True',
                                                            validation_split=0.2,
                                                            subset="validation",
                                                            seed=123
                                                            ) 
        print('tenemos %d imagenes de %s pixeles' %(len(datos_entrenamiento),tamaño_a,))
        #print('tenemos %d imagenes de %s pixeles' %(len(datos_entrenamiento),tamaño_a,))
        
        #time.sleep(2)
        self.historia=self.modelo.fit(datos_entrenamiento,
                                      validation_data=datos_validacion,epochs=epochs)
        # prueba_loss, prueba_acc = self.modelo.evaluate(datos_validacion, verbose=2)
        
       
    
    
    def graficos(self, historia,epocas):
        
        x=[i for i in range(epocas)]
        
        plt.figure(0)
        plt.scatter(x,self.historia.history['accuracy'],c='blue')
        plt.plot(x,self.historia.history['accuracy'],label='accuracy',color='blue')
        #plt.plot(self.historia.history['val_accuracy'],label='val_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epocas')
        plt.legend(loc='lower right')
        
        plt.ylabel('loss')
        plt.xlabel('epocas')
        plt.scatter(x,self.historia.history['loss'],c='r')
        plt.plot(x,self.historia.history['loss'],color='red',label='loss')
        plt.legend(loc='lower right')
        plt.rcParams.update({'axes.facecolor':(1.0,0.93,0.8039)})
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        
        plt.ylim(0,1.2)
        
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
        #plt.rcParams.update({'axes.facecolor':(1.0,0.93,0.8039)})
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        plt.show()
        
   
    def predicciones(self,ruta,img_size,datadir):
        categorias=[i for i in os.listdir(datadir)]
        prediccion=[]
        img_array=cv2.imread(ruta) 
        new_array=cv2.resize(img_array, (img_size,img_size))
        prediccion.append([new_array])
        prediccion=np.array(prediccion).reshape(-1,img_size,img_size, 3)
        prediccion = prediccion.astype('float32')
        prediccion /= 255 
        a=self.modelo.predict(prediccion)
        a[0]=list(a[0])
        jaj=np.argmax(a[0], axis=0)
        print('Hay %2.2f de probabilidad de que sea %s' %(max(a[0]),categorias[jaj]))
        print(a)       