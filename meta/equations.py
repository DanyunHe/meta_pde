import numpy as np

def F1(x,t):
  return np.sin(2*x)*np.sin(np.sin(200*x**(1.2)))

def F2(x,t):
  return np.sin(2*x)*np.sin(np.sin(x**(1.3)))

def F3(x,t):
  return np.sin(20*x)*np.sin(np.sin(5*x**(0.9)))

# Generate diffenert equations 
def generate_equations(idx):
  np.random.seed(idx)
  a=idx
  b=idx*0.5
  # c=np.random.rand()*4
  return lambda x,t : np.sin(a*(x-t))*np.sin(np.sin(b*(x-t)**(2)))
