# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 02:53:09 2021

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

class Conv2D:
    
    def __init__(self,datadir,NOMBRE=remove(time.asctime()[0:11])+remove(time.asctime()[13:19])):
        
        salida=len([i for i in os.listdir(datadir)])
        train_data = pickle.load(open('X_train.pickle','rb'))/255.0
        val_data = np.array(pickle.load(open('y_train.pickle','rb')))   
        tamaño=train_data.shape[1:]
        
        
        self.modelo=tf.keras.models.Sequential([
            #tf.keras.layers.Dropout(0.3, input_shape=(tamaño)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid',input_shape=tamaño),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64,(3,3),activation='sigmoid'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='sigmoid'),
            #tf.keras.layers.Dense(10, kernel_regularizer='l1'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(salida,activation='softmax')
            ])
    
        opt=tf.keras.optimizers.Adam(lr=1e-3)#,decay=1e-7)
        
        self.modelo.compile(optimizer=opt,
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])#'sparse_categorical_accuracy')
                       #
        
        
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NOMBRE))
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint("models/{}.model".format(NOMBRE, monitor='val_accuracy', 
                                                                                      verbose=1, save_best_only=True, mode='max'))
        self.modelo.save('models/{}'.format('ModeloConvolucional'+NOMBRE))#+'.h5')
        
        self.modelo.summary()
        
        
    def historia(self,epochs):
        
        train_data = pickle.load(open('X_train.pickle','rb'))/255.0
        val_data = np.array(pickle.load(open('y_train.pickle','rb')))
        
        print('tenemos %d imagenes de %d pixeles' %(len(train_data),train_data.shape[1]))
        time.sleep(2)
        
        
        
        
        
         
        
        self.historia=self.modelo.fit(train_data,
                                      val_data,
                                      batch_size=16,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_split=0.2,
                                      callbacks=[self.tensorboard])#,self.checkpoint])
        
        #try:
         #   test_data = pickle.load(open('X_test.pickle','rb'))/255.0
          #  test_label = np.array(pickle.load(open('y_test.pickle','rb')))
           # test_loss, test_acc = self.modelo.evaluate(test_data, test_label, verbose=2)
            #print(test_loss)
            #print(test_acc)
        #except Exception as e:
         #   pass
        
        


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
        plt.plot(x,self.historia.history['accuracy'],'--',label='accuracy',color='blue')
        
        plt.ylabel('accuracy')
        plt.xlabel('epocas')
        plt.legend(loc='lower right')
        
        plt.ylabel('val_accuracy')
        plt.xlabel('epocas')
        plt.scatter(x,self.historia.history['val_accuracy'],c='black')
        plt.plot(x,self.historia.history['val_accuracy'],'--',color='black',label='val_accuracy')
        plt.ylim(0,1.2)
        plt.legend(loc='lower right')
        plt.rcParams.update({'axes.facecolor':(1.0,0.93,0.8039)})
        plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        plt.show()
        
    
    def prediccion(self,ruta,img_size,datadir):
        
        categorias=[i for i in os.listdir(datadir)]
        prediccion=[]
        img_array=cv2.imread(ruta)#, cv2.IMREAD_GRAYSCALE) 
        new_array=cv2.resize(img_array, (img_size,img_size))
        prediccion.append([new_array])
        prediccion=np.array(prediccion).reshape(-1,img_size,img_size, 3)#1)
        prediccion = prediccion.astype('float32')
        prediccion /= 255.0
        a=self.modelo.predict(prediccion)
        a[0]=list(a[0])
        jaj=np.argmax(a[0], axis=0)
        print('Hay %2.2f de probabilidad de que sea %s' %(max(a[0])*100,categorias[jaj]))
        print(a)
    
    def predicciones(self,ruta,img_size,datadir):
        
        categorias=[i for i in os.listdir(ruta)]
        predict=[]
        for cat in categorias:
            path = os.path.join(ruta,cat)
            clase_num = categorias.index(cat)
            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(path)#, cv2.IMREAD_GRAYSCALE) 
                    new_array=cv2.resize(img_array, (img_size,img_size))
                    prediccion.append([new_array])
                    prediccion=np.array(prediccion).reshape(-1,img_size,img_size, 3)#1)
                    prediccion = prediccion.astype('float32')
                    prediccion /= 255.0
                    a=self.modelo.predict(prediccion)
                    jaj=a[0].index(max(a[0]))
                    predict.append([clase_num,jaj])
                except Exception as e:
                    pass
        
        return predict
        
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
        os.remove('X_train.pickle')
        os.remove('y_train.pickle')
    except Exception as e:
        pass

    for caracteristicas,etiqueta in train_data:
        X.append(caracteristicas)
        y.append(etiqueta)
    X=np.array(X).reshape(-1, img_size,img_size, 3)#1) #El uno hace referencia al canal de color
    
    pickle_out=open('X_train.pickle', 'wb')
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out=open('y_train.pickle', 'wb')
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
    
    
    return categorias#,train_data,val_data,tamaño

def creador_datos_evaluacion(datadir_eval,img_size): #si es que ya hay una carpeta test
    a=img_size
    categorias=[i for i in os.listdir(datadir_eval)]
    test_data = []
    X= []  #Conjunto de caracteristicas
    y= []
    
    for cat in categorias:
        path = os.path.join(datadir_eval,cat) #/placas/categoria/
        clase_num = categorias.index(cat)
        for img in os.listdir(path): #/placas/categoria/imagen
            try:
                img_array=cv2.imread(os.path.join(path,img))#, cv2.IMREAD_GRAYSCALE) 
            #reducimos a una escala de grises
                new_array=cv2.resize(img_array, (a,a))
                test_data.append([new_array,clase_num])
            except Exception as e:
                pass
    random.shuffle(test_data)
    
    try:
        os.remove('X_test.pickle')
        os.remove('y_test.pickle')
    except Exception as e:
        pass

    for caracteristicas,etiqueta in test_data:
        X.append(caracteristicas)
        y.append(etiqueta)
    X=np.array(X).reshape(-1, img_size,img_size, 3)#1) #El uno hace referencia al canal de color
    
    pickle_out=open('X_test.pickle', 'wb')
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out=open('y_test.pickle', 'wb')
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
    
    
    return categorias#,train_data,val_data,tamaño
def creador_datos_evaluacion_entrenamiento(datadir,img_size,test_size): #si es que ya hay una carpeta test
    
    a=img_size
    categorias=[i for i in os.listdir(datadir)]
    train_data=[]
    test_data = []
    X_train= []  #Conjunto de caracteristicas
    y_train= []
    X_test= []  #Conjunto de caracteristicas
    y_test= []
    
    
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
    
    test_data=[train_data.pop(i) for i in range(int(len(train_data[1])*test_size/100))]
    
    try:
        os.remove('X_train.pickle')
        os.remove('y_train.pickle')
        os.remove('X_test.pickle')
        os.remove('y_test.pickle')
    except Exception as e:
        pass

    for caracteristicas,etiqueta in train_data:
        X_train.append(caracteristicas)
        y_train.append(etiqueta)
    X_train=np.array(X_train).reshape(-1, img_size,img_size, 3)#1) #El uno hace referencia al canal de color
    
    for caracteristicas,etiqueta in test_data:
        X_test.append(caracteristicas)
        y_test.append(etiqueta)
    X_test=np.array(X_test).reshape(-1, img_size,img_size, 3)#1)
    
    pickle_out=open('X_train.pickle', 'wb')
    pickle.dump(X_train,pickle_out)
    pickle_out.close()

    pickle_out=open('y_train.pickle', 'wb')
    pickle.dump(y_train,pickle_out)
    pickle_out.close()
    
    pickle_out=open('X_train.pickle', 'wb')
    pickle.dump(X_test,pickle_out)
    pickle_out.close()

    pickle_out=open('y_ttrain.pickle', 'wb')
    pickle.dump(y_test,pickle_out)
    pickle_out.close()
    
    
    
    return categorias#,train_data,val_data,tamaño
