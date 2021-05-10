import numpy as np

def F1(x,t):
  return np.sin(2*x)*np.sin(np.sin(200*x**(1.2)))

def F2(x,t):
  return np.sin(2*x)*np.sin(np.sin(x**(1.3)))

def F3(x,t):
  return np.sin(20*x)*np.sin(np.sin(5*x**(0.9)))

def generate_equations(idx):
	a=idx
	b=idx*0.5
	return lambda x,t : np.sin(a*(x-t))*np.sin(np.sin(b*(x-t)**(0.9)))