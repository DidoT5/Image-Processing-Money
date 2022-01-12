# -*- coding: utf-8 -*-
"""
Funciones auxiliares: cargar, mostrar e imágenes por defecto.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import glob


def muestraImagen(img):
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    
def cargaImagen(img):
    return cv.cvtColor(cv.imread(img), cv.COLOR_BGR2RGB)

def cargaImagenGray(img):
    return cv.cvtColor(cv.imread(img), cv.COLOR_RGB2GRAY)

def muestraMultiplesImagenes(imgs):
    _, ax = plt.subplots(1,len(imgs),figsize=(20,20))
    for img,ax in zip(imgs,ax):
        ax.imshow(img)
        ax.axis('off')
    plt.show()


# imgs es una lista de listas []
def muestraMultiplesImagenesCuadradoColor(imgs,columns,rows):
    fig = plt.figure(figsize=(17, 17))
    columns =5
    rows = 5

    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i-1])

    plt.show()

# imgs es una lista de listas []
def muestraMultiplesImagenesCuadradoGray(imgs,columns,rows):
    fig = plt.figure(figsize=(17, 17))

    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i-1], cmap="gray")

    plt.show()

def muestraMultiplesImagenesGray(imgs):
    _, ax = plt.subplots(1,len(imgs),figsize=(17,17))
    for img,ax in zip(imgs,ax):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()
    
# dada la ruta del archivo de un billete o template, retorna un 
# número correspondiente con su valor

def name(nombre):
    if("cinco" in nombre):
        return 5
    elif("diez" in nombre):
        return 10
    elif("veinte" in nombre):
        return 20
    elif("cincuenta" in nombre):
        return 50
    else:
        return 100

def cargarTemplates(route="./templates/*.png"):
    files = glob.glob(route)
    correspondencia = [name(nombre) for nombre in files]
    templates = [cargaImagen(template) for template in files]
    return templates, correspondencia

# CARGA DE IMÁGENES POR DEFECTO #
cinco = cargaImagen("Imagenes/billete5.png")
diez = cargaImagen("Imagenes/billete10.png")
veinte = cargaImagen("Imagenes/billete20.png")
cincuenta = cargaImagen("Imagenes/billete50.png")
cien = cargaImagen("Imagenes/billete100.jpg")
cincor = cargaImagen("Imagenes/billete5reverso.png")
diezr = cargaImagen("Imagenes/billete10reverso.png")
veinter = cargaImagen("Imagenes/billete20reverso.png")
cincuentar = cargaImagen("Imagenes/billete50reverso.png")
cienr = cargaImagen("Imagenes/billete100reverso.png")
cincoReal = cargaImagen("Imagenes/cincoReal.jpeg")
cincoRealr = cargaImagen("Imagenes/cincoRealr.jpeg")
veinteReal = cargaImagen("Imagenes/veinteReal.jpeg")
veinteRealr = cargaImagen("Imagenes/veinteRealr.jpeg")
    