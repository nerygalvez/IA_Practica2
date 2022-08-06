#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

#source = None

PORCENTAJE_SLICE = 0.8 #Porcentaje a tomar para el conjunto de entrenamiento

"""
*   Devuelve un arreglo de imágenes
*   @carpeta_imagenes = ruta donde se encuentran las imágenes a cargar
"""
def cargarImagenes(carpeta_imagenes):
    #carpeta_imagenes = "Dataset_Escudos/USAC"
    imagenes = []
    for nombre_imagen in os.listdir(carpeta_imagenes):
        img = plt.imread(os.path.join(carpeta_imagenes, nombre_imagen))
        if img is not None:
                img = np.array(img, dtype = float)
                temp = img
                #Linealizo la imagen
                img = img.reshape(-1)

                #Verifico que tenga el valor que quiero porque sino no funciona y se arruina el shape
                # El shape esperado es (128, 128, 3)
                # 49152 = 128 x 128 x 3
                # Las imágenes de 128 x 128 y en cada pixel hay 3 elementos (RGB)
                if img.shape[0] != 49152:
                    print("Error en imagen -> img.shape = ", img.shape, " --- ", temp.shape, ", Ruta: ", carpeta_imagenes + "/" + nombre_imagen)
                    pass
                else:
                    imagenes.append(img) #Agrego la imagen al arreglo
                
                
                
                

    
    #print('Shape imagenes: ', np.array(imagenes).shape)
    return imagenes


"""
*   Función que separa un arreglo en dos, uno para el conjunto de entrenamiento y el otro el conjunto de validación
*   Toma el PORCENTAJE_SLICE del arreglo para el conjunto de entrenamiento y el resto para el conjunto de validación
"""
def separarArreglo(arreglo):
    slice_point = int(len(arreglo) * PORCENTAJE_SLICE)
    entrenamiento = arreglo[0 : slice_point] #No toma la posición de slice_point
    evaluacion = arreglo[slice_point : ]



    return entrenamiento, evaluacion

#result = np.array(data)
#np.random.shuffle(result)

"""
*   Función que devuelve los datos para el modelo de la USAC
"""
def load_dataset(ruta1, ruta2, ruta3, ruta4):

    #Obtengo el arreglo de todas las imágenes
    #arreglo_usac = cargarImagenes("Dataset_Escudos/USAC")
    #arreglo_landivar = cargarImagenes("Dataset_Escudos/Landivar")
    #arreglo_mariano = cargarImagenes("Dataset_Escudos/Mariano")
    #arreglo_marroquin = cargarImagenes("Dataset_Escudos/Marroquin")

    #Tendría que cambiar los nombres de las variables para ya no distiguir USAC, Landivar, etc
    #Debería de poner un nombre general como: arreglo_model_actual

    arreglo_usac = cargarImagenes(ruta1)
    arreglo_landivar = cargarImagenes(ruta2)
    arreglo_mariano = cargarImagenes(ruta3)
    arreglo_marroquin = cargarImagenes(ruta4)

    #Creo el arreglo de respuestas correctas
    respuestas_usac = [1 for _ in range(len(arreglo_usac))]
    respuestas_landivar = [0 for _ in range(len(arreglo_landivar))]
    respuestas_mariano = [0 for _ in range(len(arreglo_mariano))]
    respuestas_marroquin = [0 for _ in range(len(arreglo_marroquin))]

    #Debería de crear un arreglo de entrenamiento, un arreglo de respuestas y darles shuffle, pero se perderían las respuestas

    #Tomo el conjunto de entrenamiento y el conjunto de validación de todos los arreglos
    entrenamiento_usac, validacion_usac = separarArreglo(arreglo_usac)
    respuesta_entrenamiento_usac, respuesta_validacion_usac = separarArreglo(respuestas_usac)

    entrenamiento_landivar, validacion_landivar = separarArreglo(arreglo_landivar)
    respuesta_entrenamiento_landivar, respuesta_validacion_landivar = separarArreglo(respuestas_landivar)

    entrenamiento_mariano, validacion_mariano = separarArreglo(arreglo_mariano)
    respuesta_entrenamiento_mariano, respuesta_validacion_mariano = separarArreglo(respuestas_mariano)

    entrenamiento_marroquin, validacion_marroquin = separarArreglo(arreglo_marroquin)
    respuesta_entrenamiento_marroquin, respuesta_validacion_marroquin = separarArreglo(respuestas_marroquin)

    #Uno cada conjunto en un arreglo final
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = [], [], [], []

    train_set_x_orig.extend(entrenamiento_usac)
    train_set_x_orig.extend(entrenamiento_landivar)
    train_set_x_orig.extend(entrenamiento_mariano)
    train_set_x_orig.extend(entrenamiento_marroquin)

    train_set_y_orig.extend(respuesta_entrenamiento_usac)
    train_set_y_orig.extend(respuesta_entrenamiento_landivar)
    train_set_y_orig.extend(respuesta_entrenamiento_mariano)
    train_set_y_orig.extend(respuesta_entrenamiento_marroquin)

    test_set_x_orig.extend(validacion_usac)
    test_set_x_orig.extend(validacion_landivar)
    test_set_x_orig.extend(validacion_mariano)
    test_set_x_orig.extend(validacion_marroquin)

    test_set_y_orig.extend(respuesta_validacion_usac)
    test_set_y_orig.extend(respuesta_validacion_landivar)
    test_set_y_orig.extend(respuesta_validacion_mariano)
    test_set_y_orig.extend(respuesta_validacion_marroquin)

    #Convierto a np.array
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = np.array(train_set_x_orig), np.array(train_set_y_orig), np.array(test_set_x_orig), np.array(test_set_y_orig) 
    
    #Les aplica reshape, convierte al arreglo en un arreglo de areglos
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


    print("--------------- Shapes obtenidos ---------------")
    print(train_set_x_orig.shape)
    print(train_set_y_orig.shape)
    print(test_set_x_orig.shape)
    print(test_set_y_orig.shape)
    print("------------------------------------------------")
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No USAC', 'USAC']
