from nn_architecture import Conv1d_Model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from equations import generate_equations


def F1(x,t):
  return np.sin(2*x)*np.sin(np.sin(200*x**(1.2)))

def F2(x,t):
  return np.sin(2*x)*np.sin(np.sin(x**(1.3)))

def F3(x,t):
  return np.sin(20*x)*np.sin(np.sin(5*x**(0.9)))

def generate_snapshots(duration=1,num_frame=10000,length=1,n=512,func=F1):
  snapshots = []
  x=np.linspace(0,length,n)
  for t in np.linspace(0.01,duration,num_frame):
    snapshots.append(func(x,t))
  snapshots = np.array(snapshots)
  return snapshots

def train(model,train_loader, optimizer, epoch):
  
  model.train()
  #### TODO copy weights from self.weights
  #### Other copies the same
  weights_copy=model.weights2
  bias_copy=model.bias
  weights_linear_copy=model.weights_linear
  bias_linear_copy=model.bias_linear

  
  ## train loader should give me like 5 examples from A task

  ## Copy model weights
  ## for num step in range(inner_steps):
  ##    do inner update
  ##    xxx
  ## obtain the final loss and .backward and optimizer.step()
  #import pdb
  #pdb.set_trace()
  inner_loop_lr=0.01
  for it,data in enumerate(train_loader): ## only 1 batch
    for num_steps in range(10):
      data_input = data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)
      target = data.reshape(-1,32,16)[:,:,1:]
      data_input, target = data_input.cuda(),target.cuda()
      output = model.forward(data_input, weights_copy=weights_copy, bias_copy=bias_copy, weights_linear_copy=weights_linear_copy, bias_linear_copy=bias_linear_copy)
      loss = nn.MSELoss(reduction='mean')(output, target.reshape(-1,480))
      #### inner loop update
      create_g=False
      retain_g=True
      grads=torch.autograd.grad(loss,weights_copy,create_graph=create_g,retain_graph=retain_g)
      for w,g in zip(weights_copy,grads):
          w=w-inner_loop_lr*g
      grads = torch.autograd.grad(loss, bias_copy,create_graph=create_g,retain_graph=retain_g) 
      for w,g in zip(bias_copy,grads):
          w=w-inner_loop_lr*g
      grads = torch.autograd.grad(loss, weights_linear_copy,create_graph=create_g,retain_graph=retain_g) ## for each parameter
      for w,g in zip(weights_linear_copy,grads):
          w=w-inner_loop_lr*g
      grads = torch.autograd.grad(loss,bias_linear_copy,create_graph=create_g, retain_graph=retain_g)
      for w,g in zip(bias_linear_copy,grads):
          w=w-inner_loop_lr*g


    optimizer.zero_grad()
    # loss = nn.MSELoss(reduction='sum')(output, target.reshape(-1,480))
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().numpy()


def test(model, test_loader):
  model.eval()
  with torch.no_grad():
    for data in test_loader:
      data_input = data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)
      target = data.reshape(-1,32,16)[:,:,1:]
      data_input, target = data_input.cuda(),target.cuda()
      prediction=model(data_input)
      np.savetxt('prediction_test',model(data_input).cpu())
      np.savetxt('./test_data',data.reshape(32,-1))
      #import pdb
      #pdb.set_trace()
      loss = nn.MSELoss(reduction='mean')(prediction, target.reshape(-1,480))
      return loss.cpu().numpy()
  

def load_all_tasks():
  data=[]
  for i in range(20):
    F=generate_equations(i)
    print(F(1,1))
    data.append(generate_snapshots(func=F))

  return np.array(data)

if __name__ == '__main__':
  model = Conv1d_Model()
  model.to(device="cuda")

  #### Prepare 20 tasks of different F. Saving data as (10000,512) arrays
  #### Then load all data as (20,10000,512) arrays
  data = load_all_tasks()
  #### The last two dimensions of data should be (32,16)
  #### eg, data = data.reshape(-1,32,16)

  optimizer = optim.Adam(model.parameters(), lr = 1e-2)
  # select_train = np.random.choice(range(len(data)), int(0.8*len(data)),replace=False)
  # select_test = np.setdiff1d(range(len(data)), select_train)
  train_kwargs = {'batch_size': 32}
  test_kwargs = {'batch_size': 32}

  ### Every time during training, we should first sample several tasks to make the train loader
  ### for test loader, we need to re-sample several other tasks
  # train_loader = torch.utils.data.DataLoader(data[select_train], **train_kwargs)
  # test_loader = torch.utils.data.DataLoader(data[select_test], **test_kwargs)

  for step in range(1, 5001):
    #### Use step instead of epoch. No well-defined epoch.
    sample_train_task = np.random.choice(range(18)) # select a task out of 20 available tasks
    ## for each task, sample 10 or more snapshots 
    sample_train_snapshot = np.random.choice(10000,10)
    train_data = data[sample_train_task][sample_train_snapshot]
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    loss=train(model, train_loader, optimizer, step)
    if step%100==0:
        sample_test_task = 18+np.random.choice(range(2)) # select a task out of 20 available tasks
        sample_test_snapshot = np.random.choice(10000,10)
        test_data = data[sample_test_task][sample_test_snapshot]
        test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
        test_loss=test(model, test_loader)
        print("step: ",step,"train loss ",loss,"test loss: ",test_loss)


  # meta_system = ExperimentBuilder(model=model,data=data)
  # meta_system.run_experiment()
