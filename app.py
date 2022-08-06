#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask,render_template,request,jsonify,redirect, abort, send_from_directory, url_for
import csv
import matplotlib.pyplot as chart

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import USAC
import Landivar
import Mariano
import Marroquin

app = Flask(__name__)


#modelo_usac = USAC.obtenerModelo()
#modelo_landivar = Landivar.obtenerModelo()
#modelo_mariano = Mariano.obtenerModelo()
#modelo_marroquin = Marroquin.obtenerModelo()
modelo_usac = None
modelo_landivar = None
modelo_mariano = None
modelo_marroquin = None


"""
*   Función para cargar la página principal de la aplicación
"""
@app.route('/')
def index():
    return render_template('index.html')




"""
*   Ruta para cargar los modelos
"""
@app.route('/obtenerModelos', methods=["POST"])
def obtenerModelos():

    #Obtengo los modelos si aún no los he obtenido
    global modelo_usac
    global modelo_landivar
    global modelo_mariano
    global modelo_marroquin

    if modelo_usac == None:
        modelo_usac = USAC.obtenerModelo()

    if modelo_landivar == None:
        modelo_landivar = Landivar.obtenerModelo()
    
    if modelo_mariano == None:
        modelo_mariano = Mariano.obtenerModelo()

    if modelo_marroquin == None:
        modelo_marroquin = Marroquin.obtenerModelo()


    #Devuelvo una respuesta cualquiera
    return jsonify(
                    estado=200,
                    mensaje='Modelos generados con éxito',
                   )








arreglo_imagenes = []

"""
*   Ruta que recibe una imagen, la almacena en el servidor y agrega la ruta
*   donde se encuentra la imagen a un arreglo, esto para usar las imágenes posteriormente
*   en la función de analizar
"""
@app.route('/enviarImagen', methods=["POST"])
def enviarImagen():

    print("hola")
    uploaded_files = request.files.getlist("archivo1")

    #print(uploaded_files)
    
    for f in uploaded_files:
        #Obtengo la imagen y la guardo en el servidor
        filename = f.filename
        #print('Nombre: ', filename)
        #f.save("uploads/" + filename)
        f.save("static/" + filename)
        #arreglo_imagenes.append("uploads/" + filename)
        arreglo_imagenes.append("static/" + filename)
    
    print(arreglo_imagenes)
    #print(request.files)
    print("mundo")
    return jsonify(
                    estado=200,
                    mensaje='Haciendo pruebas',
                   )

    #Obtengo la imagen y la guardo en el servidor
    f = request.files['archivo1']
    filename = f.filename
    #print('Nombre: ', filename)
    #f.save("uploads/" + filename)
    f.save("static/" + filename)

    #arreglo_imagenes.append("uploads/" + filename)
    arreglo_imagenes.append("static/" + filename)
    
    print(arreglo_imagenes)

    #Pruebo leer la imagen del servidor
    #img = plt.imread("uploads/" + filename)
    #Debería de verificar el shape de la imagen para que no tire error
    #img = np.array(img, dtype = float)
    #Linealizo la imagen
    #img = img.reshape(-1)
    #print('Shape: ', img.shape)


    #Devuelvo una respuesta cualquiera
    return jsonify(
                    estado=200,
                    mensaje='La operación de recibir la imagen se realizó con éxito',
                   )



"""
*   Ruta que vacía el contenido del arreglo arreglo_imagenes
"""
@app.route('/eliminarImagenes', methods=["POST"])
def eliminarImagenes():
    arreglo_imagenes.clear()
    #arreglo_imagenes = []

    #Devuelvo una respuesta cualquiera
    return jsonify(
                    estado=200,
                    mensaje='La operación de eliminar imágenes se realizó con éxito',
                   )


"""
*   Ruta que analiza el conjunto de imágenes
"""
@app.route('/analizar', methods=["POST"])
def analizar():

    print('***** Imágenes a analizar *****')
    print(arreglo_imagenes)

    #Siempre voy a devolver resultados1 y resultados2
    resultados1 = []
    resultados2 = []
    total_usac = 0
    total_landivar = 0
    total_mariano = 0
    total_marroquin = 0
    aciertos_usac = 0
    aciertos_landivar = 0
    aciertos_mariano = 0
    aciertos_marroquin = 0

    if len(arreglo_imagenes) < 6: #Se cargaron menos de 6 imágenes

        for i in range(0, len(arreglo_imagenes)):
            # Prueba de prediccion
            img = plt.imread(arreglo_imagenes[i])
            
            #Debería de verificar el shape de la imagen para que no tire error
            img = np.array(img, dtype = float)
            img = img.reshape(-1) #Linealizo la imagen

            if img.shape[0] != 49152: #La imagen no es valida
                print("Error en imagen -> img.shape = ", img.shape, " --- ", ", Ruta: ", arreglo_imagenes[i])
                pass
            else: #La imagen es valida
                
                p = [255]
                for e in img:
                    p.append(e)
                
                p = np.array(p)
                
                resultados = []
                resultados.append({"institucion" : "USAC", "probabilidad" : modelo_usac.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})
                resultados.append({"institucion" : "Landivar", "probabilidad" : modelo_landivar.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})
                resultados.append({"institucion" : "Mariano", "probabilidad" : modelo_mariano.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})
                resultados.append({"institucion" : "Marroquin", "probabilidad" : modelo_marroquin.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})

                #print('***** RESULTADOS *****')
                #print(resultados)
                #rint('')

                mejor_resultado = sorted(resultados, key=lambda item: item['probabilidad'], reverse=True)[:1] #Los ordena de mayor a menor
                print('Mejor resultado', mejor_resultado[0])

                resultados1.append(mejor_resultado[0])

                
        
    else: #Se cargaron de 6 a N imágenes
        #Tengo que splitear el nombre de la imagen para ver la respuesta correcta
        #y así generar el informe final

        for i in range(0, len(arreglo_imagenes)):

            #El nombre de la imagen viene de la forma: static/institucion_numeroImagen.jpg
            respuestaCorrecta = arreglo_imagenes[i].split("/")[1].split("_")[0].lower()
            print('respuestaCorrecta: ', respuestaCorrecta)

            # Prueba de prediccion
            img = plt.imread(arreglo_imagenes[i])
            
            #Debería de verificar el shape de la imagen para que no tire error
            img = np.array(img, dtype = float)
            img = img.reshape(-1) #Linealizo la imagen

            if img.shape[0] != 49152: #La imagen no es valida
                print("Error en imagen -> img.shape = ", img.shape, " --- ", ", Ruta: ", arreglo_imagenes[i])
                pass
            else: #La imagen es valida
                
                p = [255]
                for e in img:
                    p.append(e)
                
                p = np.array(p)
                
                resultados = []
                resultados.append({"institucion" : "USAC", "probabilidad" : modelo_usac.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})
                resultados.append({"institucion" : "Landivar", "probabilidad" : modelo_landivar.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})
                resultados.append({"institucion" : "Mariano", "probabilidad" : modelo_mariano.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})
                resultados.append({"institucion" : "Marroquin", "probabilidad" : modelo_marroquin.predecir_probabilidad(p), "imagen" : arreglo_imagenes[i]})

                #print('***** RESULTADOS *****')
                #print(resultados)
                #rint('')

                mejor_resultado = sorted(resultados, key=lambda item: item['probabilidad'], reverse=True)[:1] #Los ordena de mayor a menor
                print('Mejor resultado', mejor_resultado[0])

                #resultados2.append(mejor_resultado[0])
                # Obtengo la respuesta del modelo y lo paso a minúscula  
                respuestaModelo = mejor_resultado[0]['institucion'].lower()   

                #Verifico cuál es la respuesta correcta, ya pasé todo a minúscula
                if respuestaCorrecta == "usac":
                    total_usac += 1 #Aumento en uno la cantidad de imágenes de esta universidad
                    if respuestaCorrecta == respuestaModelo: #Si el modeló acertó
                        aciertos_usac += 1 #Aumento en uno el número de aciertos del modelo

                elif respuestaCorrecta == "landivar":
                    total_landivar += 1 #Aumento en uno la cantidad de imágenes de esta universidad
                    if respuestaCorrecta == respuestaModelo: #Si el modeló acertó
                        aciertos_landivar += 1 #Aumento en uno el número de aciertos del modelo
                
                elif respuestaCorrecta == "mariano":
                    total_mariano += 1 #Aumento en uno la cantidad de imágenes de esta universidad
                    if respuestaCorrecta == respuestaModelo: #Si el modeló acertó
                        aciertos_mariano += 1 #Aumento en uno el número de aciertos del modelo
                
                elif respuestaCorrecta == "marroquin":
                    total_marroquin += 1 #Aumento en uno la cantidad de imágenes de esta universidad
                    if respuestaCorrecta == respuestaModelo: #Si el modeló acertó
                        aciertos_marroquin += 1 #Aumento en uno el número de aciertos del modelo





    print('resultados1')
    print(resultados1)

    #Le doy forma a resultados2
    resultados2 = [
                    {"usac" : (aciertos_usac * 100 / total_usac) if total_usac != 0 else 0}
                    ,{"landivar" : (aciertos_landivar * 100 / total_landivar) if total_landivar != 0 else 0}
                    ,{"mariano" : (aciertos_mariano * 100 / total_mariano) if total_mariano != 0 else 0}
                    ,{"marroquin" : (aciertos_marroquin * 100 / total_marroquin) if total_marroquin != 0 else 0}
                ]
    print('resultados2')
    print(resultados2)

    #resultados1 = [{"a":4, "b": "hola", "c": "valor"}]

    print(len(resultados1))
    print(len(resultados2))
    #Devuelvo
    return jsonify(
                    estado=200,
                    resultados1=resultados1,
                    resultados2=resultados2,
                   )



"""
*   Ruta para ejecutar el servidor de Flask
"""
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

    