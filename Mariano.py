#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
*   MODELO QUE APRENDE CUÁL ES UN ESCUDO DE LA Mariano
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter

"""
*   Muentra en pantalla la imagen que se encuentra en la ruta especificada
*   @ruta = ruta donde se encuentra la imagen a mostrar
"""
def mostrarImagen(ruta):
    img = plt.imread("Dataset_Escudos/USAC/1.jpg")
    #print(len(img[0][0]))
    plt.imshow(img)
    plt.show()

#Imprimiendo la imagen por fila


"""
*   Función que evalúa 5 posibles modelos para el reconocimiento del escudo de la USAC
"""
def entrenarModelos():

    #Cargando conjuntos de datos
    ruta1 = "Dataset_Escudos/USAC"
    ruta2 = "Dataset_Escudos/Landivar"
    ruta3 = "Dataset_Escudos/Mariano"
    ruta4 = "Dataset_Escudos/Marroquin"
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File.load_dataset(ruta3, ruta1, ruta2, ruta4)

    # Convertir imagenes a un solo arreglo
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    # Vean la diferencia de la conversion
    #print('Original: ', train_set_x_orig.shape)
    #print('Con reshape: ', train_set_x.shape)

    # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)

    # Se entrenan los modelos
    model1 = Model(train_set, test_set, reg=False, alpha=0.001, lam=0, MAX_ITERATIONS = 500, MIN_VALUE = 0.1, STEP = 50)
    model1.training()
    
    model2 = Model(train_set, test_set, reg=True, alpha=0.0005, lam=150, MAX_ITERATIONS = 100, MIN_VALUE = 0.1, STEP = 50)
    model2.training()

    model3 = Model(train_set, test_set, reg=False, alpha=0.0025, lam=0, MAX_ITERATIONS = 350, MIN_VALUE = 0.1, STEP = 50)
    model3.training()

    model4 = Model(train_set, test_set, reg=True, alpha=0.000009, lam=10, MAX_ITERATIONS = 1000, MIN_VALUE = 0.1, STEP = 50)
    model4.training()

    model5 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=0, MAX_ITERATIONS = 250, MIN_VALUE = 0.1, STEP = 50)
    model5.training()

    # Se grafican los entrenamientos
    Plotter.show_Model([model1, model2, model3, model4, model5])


    # Prueba de prediccion
    #img = plt.imread("Dataset_Escudos/USAC/1.jpg")
    #img = plt.imread("Dataset_Escudos/Landivar/1.jpg")
    #img = plt.imread("Dataset_Escudos/Mariano/1.jpg")
    #img = plt.imread("Dataset_Escudos/Marroquin/1.jpg")
    
    #Debería de verificar el shape de la imagen para que no tire error
    #img = np.array(img, dtype = float)
    #img = img.reshape(-1) #Linealizo la imagen

    #p = [255]
    #for e in img:
    #    p.append(e)
    
    #p = np.array(p)

    #result = model1.predict(p)
    #print('--', classes[result[0]], '--')

"""
*   Función que devuelve el modelo que fue considerado el mejor dentro de los 5 modelos entrenados
"""
def obtenerModelo():
    #Cargando conjuntos de datos
    ruta1 = "Dataset_Escudos/USAC"
    ruta2 = "Dataset_Escudos/Landivar"
    ruta3 = "Dataset_Escudos/Mariano"
    ruta4 = "Dataset_Escudos/Marroquin"
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File.load_dataset(ruta3, ruta1, ruta2, ruta4)

    # Convertir imagenes a un solo arreglo
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    # Vean la diferencia de la conversion
    #print('Original: ', train_set_x_orig.shape)
    #print('Con reshape: ', train_set_x.shape)

    # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)
    
    #La decisión es entre el modelo 1 y el 2
    model1 = Model(train_set, test_set, reg=False, alpha=0.001, lam=0, MAX_ITERATIONS = 500, MIN_VALUE = 0.1, STEP = 50)
    model1.training()
    
    #model2 = Model(train_set, test_set, reg=True, alpha=0.0005, lam=150, MAX_ITERATIONS = 100, MIN_VALUE = 0.1, STEP = 50)
    #model2.training()

    return model1
    #return model2



#entrenarModelos()