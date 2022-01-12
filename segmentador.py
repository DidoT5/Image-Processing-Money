#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import imutils

def orderPoints(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def fourPointTransform(image, pts):
    
    rect = orderPoints(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def detectorDeBilletes(image,show):
    res = []

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 75, 200)

    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    dilated = cv.dilate(edged, kernel, iterations=3)

    cnts = cv.findContours(dilated.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    rect_points = []
    section_points = []

    for c in cnts:

        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect = orderPoints(approx.reshape(4,2))
            (a1,b1,c1,d1) = rect
            existe = False
            for rc in rect_points:
                (a2,b2,c2,d2) = rc
                a = np.sqrt(((a1[0] - a2[0]) ** 2) + ((a1[1] - a2[1]) ** 2)) <= 10
                b = np.sqrt(((b1[0] - b2[0]) ** 2) + ((b1[1] - b2[1]) ** 2)) <= 10
                c = np.sqrt(((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2)) <= 10
                d = np.sqrt(((d1[0] - d2[0]) ** 2) + ((d1[1] - d2[1]) ** 2)) <= 10
                if sum([a,b,c,d]) >=3:
                    existe = True
                    break
            if not existe:
                rect_points += [rect]
                section_points += [approx]

    cp = image.copy()

    for sc in section_points:
        cv.drawContours(cp, [sc], -1, (0, 255, 0), 2)
    
    if(show):
        cv.imshow('Contours', cp)
        cv.waitKey()
        cv.destroyAllWindows()

    for rc in rect_points:
        warped = fourPointTransform(image, rc.reshape(4, 2))
        res.append(warped.copy())
    return res
        
def detectorDeMonedas(image,show):
    res = []
    rect_points = []
    
    cp = image.copy()
    gray = cv.cvtColor(cp, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 75, 200)
    
    kernel = np.array([[0,0,0],[1,1,1],[0,0,0]],np.uint8)
    dilated = cv.dilate(edged, kernel, iterations=2)

    kernel = np.ones((3, 3), np.uint8)

    closing = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel, iterations=4)
    cont_img = closing.copy()
    contours, hierarchy = cv.findContours(cont_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) < 5:
            continue
        ellipse = cv.fitEllipse(cnt)
        ((x,y),(w,h),_) = ellipse
        if(abs(w-h)<20 and w>20 and h>20):
            a = [int(x-w/2),int(y-h/2)]
            b = [int(x+w/2),int(y+h/2)]
            c = [int(x-w/2),int(y+h/2)]
            d = [int(x+w/2),int(y-h/2)]
            pts = np.array([a,b,c,d])
            rect_points.append(pts)
            cv.ellipse(cp, ellipse, (0,255,0), 2)
            
    for rc in rect_points:
        warped = fourPointTransform(image, rc)
        res.append(warped.copy())
    if(show):
        cv.imshow('Contours', cp)
        cv.waitKey()
        cv.destroyAllWindows()
    return res

def analizaFoto(image,show=False):
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    billetes = detectorDeBilletes(image,show)
    monedas = detectorDeMonedas(image,show)
    res = billetes + monedas
    if(show):
        for money in res:
            cv.imshow("Money", money)
            cv.waitKey()
        cv.destroyAllWindows()
    return billetes, monedas