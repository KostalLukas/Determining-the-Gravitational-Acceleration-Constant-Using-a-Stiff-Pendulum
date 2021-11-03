import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp
import sympy as sym

t_s = np.loadtxt('Pendulum Data.csv', #loads 20 values for t_s in s
                                  usecols=(0), skiprows=(1), max_rows=(20),delimiter=',', unpack=True)

t_p, t_r= np.loadtxt('Pendulum Data.csv', #loads 10 values for each t_p and t_r in s
                                 usecols=(1, 2), skiprows=(1), max_rows=(10), delimiter=',', unpack=True)

valM_p = 0.74678 #mass of stiff pendulum in Kg
valM_r = 0.26832 #mass of uniform cylindrical rod in Kg
errM = 1e-5 #uncertainty in measurement of mass in Kg


valh = 0.532 #distance from knife edge to centre of mass in m
valL = 0.395 #length of the uniform cylindrical rod in m
errh = 2e-3 #uncertainty in measurement of distance h in m
errL = 1e-3 #uncertainty in measuremnt of length in m

valT_s = t_s/20 #converting the times for multiple oscillations into period
valT_p = t_p/5
valT_r = t_r/5

meanT_s = np.mean(valT_s) #calcualting the mean periods of oscillation
meanT_p = np.mean(valT_p)
meanT_r = np.mean(valT_r)

SerrT_s = np.std(valT_s, ddof = 1)/np.sqrt(20) #calculate standard error on the mean for periods
SerrT_p = np.std(valT_p, ddof = 1)/np.sqrt(10)
SerrT_r = np.std(valT_r, ddof = 1)/np.sqrt(10)

T_s, T_p, T_r, M_p, M_r, h, L = sym.symbols('T_s T_p T_r M_p M_r h L', real=True) #define variables for sympy

gfunc = (2*sp.pi/T_s)**2 * h * (1/12 * M_r/M_p * ((T_p*L)/(T_r*h))**2 +1) #define formula for calculating g

gwrtT_s = sym.diff(gfunc, T_s) #calculating the partial derivatives for error propagation
gwrtT_r = sym.diff(gfunc, T_r)
gwrtT_p = sym.diff(gfunc, T_p)
gwrtM_p = sym.diff(gfunc, M_p)
gwrtM_r = sym.diff(gfunc, M_r)
gwrth = sym.diff(gfunc, h)
gwrtL = sym.diff(gfunc, L)

valg = gfunc.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r, M_p:valM_p, M_r:valM_r, h:valh, L:valL}) #calculate value for g

valgwrtT_s = gwrtT_s.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r, #evaluate the partial derivatives
                           M_p:valM_p, M_r:valM_r, h:valh, L:valL})
valgwrtT_r = gwrtT_r.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r,
                           M_p:valM_p, M_r:valM_r, h:valh, L:valL})
valgwrtT_p = gwrtT_p.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r,
                           M_p:valM_p, M_r:valM_r, h:valh, L:valL})
valgwrtM_p = gwrtM_p.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r,
                           M_p:valM_p, M_r:valM_r, h:valh, L:valL})
valgwrtM_r = gwrtM_r.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r,
                           M_p:valM_p, M_r:valM_r, h:valh, L:valL})
valgwrth = gwrth.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r,
                           M_p:valM_p, M_r:valM_r, h:valh, L:valL})
valgwrtL = gwrtL.subs({T_s:meanT_s, T_p:meanT_p, T_r:meanT_r,
                           M_p:valM_p, M_r:valM_r, h:valh, L:valL})

sigmas = np.array([SerrT_s, SerrT_p, SerrT_r, errM, errM, errh, errL]) #array containing uncertainties
partials = np.array([valgwrtT_s, valgwrtT_r, valgwrtT_p, valgwrtM_p, valgwrtM_r, valgwrth, valgwrtL]) #array containing evaluated partial derivatives 

errg = np.sqrt(float(np.sum((sigmas*partials)**2))) #adding product of uncertainty and corresponding partial in quadrature

print('calculated value of g = %.3f ± %.3f ms^-2 (4sf)' %(valg, errg))












