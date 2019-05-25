#relativistic correction values, from Finn & Thorne (2000)
import numpy as np
import scipy.interpolate as itp
data = np.array([
[np.nan,-0.99,  -0.9,  -0.5,   0.0,   0.2,   0.5,    0.8,    0.9,   0.99,  0.999],
[1.000, 1.240, 1.233, 1.197, 1.143, 1.114, 1.053, 0.9144, 0.7895, 0.4148, 0.2022],
[1.001, 1.239, 1.232, 1.196, 1.142, 1.114, 1.053, 0.9140, 0.7894, 0.4154, 0.2032],
[1.002, 1.238, 1.231, 1.196, 1.141, 1.113, 1.052, 0.9137, 0.7894, 0.4160, 0.2041],
[1.005, 1.235, 1.228, 1.193, 1.139, 1.111, 1.050, 0.9126, 0.7891, 0.4177, 0.2069],
[1.010, 1.231, 1.224, 1.189, 1.135, 1.107, 1.047, 0.9109, 0.7887, 0.4207, 0.2116],
[1.020, 1.222, 1.215, 1.181, 1.127, 1.100, 1.041, 0.9076, 0.7880, 0.4263, 0.2208],
[1.050, 1.198, 1.192, 1.159, 1.108, 1.081, 1.025, 0.8988, 0.7867, 0.4434, 0.2473],
[1.100, 1.165, 1.159, 1.128, 1.080, 1.055, 1.002, 0.8876, 0.7859, 0.4701, 0.2281]])

def epsAtIsco(a):
    if a>0.999: raise ValueError('Spin too high for Finn&Thornes corrections')
    i=1
    while a >= data[0][i]:
        if a == data[0][i]:
            return data[1][i]
        i+=1
    fit = np.polyfit(data[0][i-1:i+1],data[1][i-1:i+1],1)
    return np.polyval(fit,a)

def eps(r,a):
    if a>0.999: raise ValueError('Spin too high for Finn&Thornes corrections')
    if r>1.05: raise ValueError('Radius too high for represented data from FT')
    i=1
    j=1
    while a>= data[0][i]:
        i+=1
    while r>= data[j][0]:
        j+=1
    rows = np.array([[i-1, i-1],
                     [i, i]], dtype=np.intp)
    columns = np.array([[j-1, j],
                        [j-1, j]], dtype=np.intp)
    points = list()
    for k in [j-1,j]:
        for l in [i-1,i]:
            points.append([data[k][0],data[0][l]])
    values = data[rows,columns].flatten()
    #print(points,values)
    f = itp.LinearNDInterpolator(points,values)
    return f(r,a)