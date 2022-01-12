# -*- coding: utf-8 -*-
"""
Detector de billetes. Debe haber una carpeta templates con las templates.
Los billetes de input, deberán tener un shape de (193,107), en caso contrario,
sufrirán un resize.

Las imágenes deberán de estar en ./Imagenes/
"""
import cv2 as cv
import numpy as np

from funciones_auxiliares import *

class detectorBillete():

    def __init__(self, templates, correspondencia):
        self.templates = templates
        self.correspondencia = correspondencia
        self.default_shape = (107,193)
        self.threshold = 0.8
    
    def detectar(self, billete, mostrar=True, descripcion=True):
        if (self.correspondencia == None or self.templates == None):
            return print("La correspondencia o las plantillas no fueron establecidas")
        
        realShape = billete.shape[:2]
        
        resizeBillete = cv.resize(billete,self.default_shape[::-1],interpolation = cv.INTER_AREA)
        prop = realShape[1]/self.default_shape[1]
        muestraImagen(resizeBillete)
        maximos = []
        posicionCuadrado = []
        for template in self.templates:        
            match = cv.matchTemplate(resizeBillete, template, method=cv.TM_CCORR_NORMED)
            h, w, _ = template.shape
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
            bottom_right = (max_loc[0] + w, max_loc[1] + h)
            squareInfo = dict()
            squareInfo["bottom_right"] = bottom_right
            squareInfo["max_loc"] = max_loc

            posicionCuadrado.append(squareInfo)
            maximos.append(np.max(match))
    
        maxpos = np.argmax(maximos)
        mediaResto = np.mean(maximos[0:maxpos]+maximos[maxpos+1:])
        maximo = np.max(maximos)
        
        deteccion = self.correspondencia[maxpos]
              
        if(mostrar):
            cuadrado = posicionCuadrado[maxpos]
            copiaB = resizeBillete.copy()
            max_loc = cuadrado["max_loc"]
            bottom_right = cuadrado["bottom_right"]
            
            cv.rectangle(copiaB,max_loc,bottom_right, 255, 1)
            cv.cvtColor(copiaB, cv.COLOR_BGR2RGB)
            muestraImagen(copiaB)  
    
        if (maximo<self.threshold):
            if(descripcion):
                print("No se ha superado el umbral. Valor= {}".format(maximo))
            deteccion = 11 # valor de billete que no existe como bandera
        else:
            if(descripcion):
                print("Se ha detectado billete de {}. Probabilidad: {}. Media resto: {}.".format(deteccion,maximo,mediaResto))
        return deteccion, maximo, mediaResto
