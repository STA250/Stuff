from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.mlab as mlab
def juicyfunc(c):
	def SinorCos(a,b):
		if c==1 : return a*math.cos(b)
		elif c==2 :return a*math.sin(b)
		else : return math.sqrt(a*a+b*b)
	return np.frompyfunc(SinorCos,2,1)
x=np.random.uniform(0,2*math.pi,1000)
y=np.random.uniform(0,1,1000)
u=juicyfunc(1)(y,x)
v=juicyfunc(2)(y,x)
w=juicyfunc(3)(u,v)
plt.plot(u,v,'bo')
juicy=PdfPages('juicy1.pdf')
plt.savefig(juicy,format='pdf')
juicy.close()
plt.close('all')
n, bins, patches = plt.hist(w, 50, normed=1, facecolor='red', alpha=0.5)
#print(n)
#print(sum(bins))
#print(patches[2])
#y = mlab.normpdf(bins, mu, sigma)
#plt.plot(bins, y, 'r--')
#plt.xlabel('Smarts')
#plt.ylabel('Probability')
juicy1=PdfPages('juicy2.pdf')
plt.savefig(juicy1,format='pdf')
juicy1.close()

