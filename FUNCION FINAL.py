# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 19:20:40 2022

@author: Enrique
"""

import segmentador
from detector_monedas import *
from detector_billetes import *
from funciones_auxiliares import * 


# Cargamos al clasificador de monedas
clasificadorMoneda = detectorMonedas()

# Cargamos al clasificador de billetes
templates,correspondencia = cargarTemplates()
clasificadorBillete = detectorBillete(templates, correspondencia)

'''
ESTA FUNCIÓN UNE AL SEGMENTADOR CON LOS CLASIFICADORES
'''
def contadorDinero(img, descripcion=True, show=False):
    billetes, monedas = segmentador.analizaFoto(image, show)
    
    clasifBilletes = []
    for billete in billetes:
        res, maximum, media = clasificadorBillete.detectar(self, billete, False, descripcion)
        if (res==13):
            if(descripcion):
                print("Unos de los billetes no superó el umbral")
        else:
            clasifBilletes.append(res)
        
    clasifMonedas = clasificadorMoneda.predecir(monedas)
    
    valorTotal = sum(clasifBilletes+clasifMonedas)
    
    if (descripcion):
        print("### BILLETES DETECTADOS ###")
        for valor in clasifBilletes:
            print("Se ha detectado un billete de {} euros".format(valor))
        
        print("")
        print("### MONEDAS DETECTADAS ###")
        for valor in clasifMonedas:
            print("Se ha detectado una moneda de {} euros".format(valor))
    
        print("")
        print("El valor total ha sido de: {} euros".format(valorTotal))
        
    return valorTotal
        
        
        
        
        
        
        
        