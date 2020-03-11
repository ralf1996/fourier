#from pylab import *
#
#x = mgrid[-10:10:0.02] # 这里类似于MATLAB用冒号产生步长为0.02的序列，但是语法和MATLAB不同
#n = arange(1,1000)
#
#def fourier_transform():
#    a0 = (1-exp(-pi))/pi+1
#    s=a0/2
#    for i in range(1,100,1):
#        s0 = ( (1-(-1)**i*exp(-pi))/(pi*(1+i**2))*cos(i*x)+1/pi*( (-i*(1-(-1)**i*exp(-pi)))/(1+i**2) + (1-(-1)**i)/i ) * sin(i*x) )
#        s=s+s0
#    plot(x,s,'orange',linewidth=0.6)
#    title('fourier_transform')
#    show()
#
#fourier_transform()

import symfit



from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt

def fourier_series(x, f, n=0):
   """
   Returns a symbolic fourier series of order `n`.

   :param n: Order of the fourier series.
   :param x: Independent variable
   :param f: Frequency of the fourier series
   """
   # Make the parameter objects for all the terms
   a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
   sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
   # Construct the series
   series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                    for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
   return series

x, y = variables('x, y')
w, = parameters('w')
model_dict = {y: fourier_series(x, f=w, n=3)}
print(model_dict)

# Make step function data
xdata = np.linspace(-np.pi, np.pi)
ydata = np.zeros_like(xdata)
ydata[xdata > 0] = 1
# Define a Fit object for this model and data
fit = Fit(model_dict, x=xdata, y=ydata)
fit_result = fit.execute()
print(fit_result)

# Plot the result
plt.plot(xdata, ydata)
plt.plot(xdata, fit.model(x=xdata, **fit_result.params).y, ls=':')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

































#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#
#tau = 0.045
#def fourier(x, *a):
#    ret = a[0] * np.cos(np.pi / tau * x)
#    for deg in range(1, len(a)):
#        ret += a[deg] * np.cos((deg+1) * np.pi / tau * x)
#    return ret
#
#xdata = np.linspace(-np.pi, np.pi)
#ydata = xdata**2
##ydata[xdata > 0] = 1
#
#
#
## Fit with 15 harmonics
#popt, pcov = curve_fit(fourier, xdata, ydata, [1.0] * 50)
#
## Plot data, 15 harmonics, and first 3 harmonics
#fig = plt.figure()
##ax1 = fig.add_subplot(111)
#p1, = plt.plot(xdata,ydata)
#p2, = plt.plot(xdata, fourier(xdata, *popt))
##p3, = plt.plot(xdata, fourier(xdata, popt[0], popt[1], popt[2]))
#plt.show()












#import numpy as np
#from lmfit import Minimizer, Parameters, report_fit
#
#def fcn2min(params, x, data):
#    """Model a decaying sine wave and subtract data."""
#    amp = params['amp']
#    shift = params['shift']
#    omega = params['omega']
#    decay = params['decay']
#
#    return model(x,amp,shift,omega,decay) - data
#def model(x,amp,shift,omega,decay):
#    return amp * np.sin(x*omega + shift) * np.exp(-x*x*decay)
#
#x = np.linspace(0, 15, 301)
#data = (5. * np.sin(2*x - 0.1) * np.exp(-x*x*0.025) +
#        np.random.normal(size=len(x), scale=0.2))
#
#
## create a set of Parameters
#params = Parameters()
#params.add('amp', value=10, min=0)
#params.add('decay', value=0.1)
#params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2,)
#params.add('omega', value=3.0)
#
## do fit, here with leastsq model
#minner = Minimizer(fcn2min, params, fcn_args=(x, data))
#result = minner.minimize()
#print(result.params)
#
#final = data + result.residual
#p = result.params
#final = model(x,p['amp'],p['shift'],p['omega'],p['decay'])
#
## write error report
#report_fit(result)
#
#try:
#    import matplotlib.pyplot as plt
#    plt.plot(x, data, 'k+')
#    plt.plot(x, final, 'r')
#    plt.show()
#except ImportError:
#    pass






