#Ce fichier sera alloué aux fonctions de la partie II du TP 01
import numpy as np
import matplotlib.pyplot as plt
import cv2
 
def OpenImage(img):
  I=plt.imread(img)
  L,C = I.shape[0],I.shape[1]
  return I,L,C

def Divide(img):
  I=plt.imread(img)
  R = I[:,:,0]
  G = I[:,:,1]
  B = I[:,:,2]   
  return R,G,B

def HSV(img):
  image = cv2 . imread (img, 3)
  image_HSV = cv2 . cvtColor ( image , cv2 . COLOR_RGB2HSV )
  return image_HSV

def CountPix(img):
  I,L,C = OpenImage(img)
  return L*C

def FactPix(x,y,I):
  I =I*x+y
  return I

def Func_a(img):
  I=plt.imread(img)
  log = np.log2(I)
  expo = np.exp(I)
  carre=np.pow(I,2)
  racine = np.sqrt(I)
  return log,expo,carre,racine

def Func_m(img):
  I=plt.imread(img)
  moyenne = I.mean()
  dev = I.std()
  return moyenne,dev

def Normalize(img):
  I=plt.imread(img)
  moyenne,dev = Func_m(img)
  I=(I-moyenne)/dev
  return I

def Inverse(img):
  I=plt.imread(img)
  Inv = I.max()-I
  return Inv

def CalcHist(img):
  I=plt.imread(img)
  H,b=np.histogram(I,bins=255)
  return H,b

def Threshold(img,seuil):
  I=plt.imread(img)
  x,y,z = I.shape
  T = I.reshape(x*y*z)
  T = np.array(list(map(lambda x:0 if x<= seuil else 255 ,T)))  
  return T.reshape(x,y,z)

def Func_j(img):
  I=plt.imread(img)
  plt.figure(0)
  plt.imshow(I)
  I_hist,I_bins = CalcHist(img)
  plt.figure(1)
  plt.plot(I_hist)
  Inv = Inverse(img)
  plt.figure(2)
  plt.imshow(Inv)

def Func_t(img):
  I=plt.imread(img)
  plt.figure(0)
  plt.imshow(I)
  I_Hist,I_bins = CalcHist(img)
  plt.figure(1)
  plt.plot(I_Hist)
  I_Normalize = Normalize(img)
  plt.figure(2)
  plt.imshow(I_Normalize)

def Func_f(img):
  I=plt.imread(img)
  plt.figure(0)
  plt.imshow(I)
  I_Hist,I_bins = CalcHist(img)
  plt.figure(1)
  plt.plot(I_Hist)
  I_Seuilee = Threshold(img,128)
  plt.figure(2)
  plt.imshow(I_Seuilee)

