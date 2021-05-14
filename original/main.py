from nn_architecture import Conv1d_Model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def generate_snapshots(duration,num_frame,length,n,func):
  snapshots = []
  x=np.linspace(0,length,n)
  for t in np.linspace(0.01,duration,num_frame):
    snapshots.append(func(x+t))
  snapshots = np.array(snapshots)
  return snapshots

def F1(x):
  return np.sin(2*x)*np.sin(np.sin(200*x**(1.2)))

def F2(x):
  return np.sin(2*x)*np.sin(np.sin(x**(1.3)))

def F3(x):
  return np.sin(20*x)*np.sin(np.sin(5*x**(0.9)))

def train(model,train_loader, optimizer, epoch):
  model.train()
  ## train loader should give me like 5 examples from A task
  
  ## Copy model weights
  ## for num step in range(inner_steps):
  ##    do inner update
  ##    xxx
  ## obtain the final loss and .backward and optimizer.step()
  # import pdb
  # pdb.set_trace()
  for batch_idx, data in enumerate(train_loader):
    data_input = data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)
    target = data.reshape(-1,32,16)[:,:,1:]
    data_input, target = data_input.cuda(),target.cuda()
    optimizer.zero_grad()
    output = model.forward(data_input)
    loss = nn.MSELoss(reduction='sum')(output, target.reshape(-1,480))
    loss.backward()
    optimizer.step()
  print(loss)

def test(model, test_loader):
  model.eval()
  with torch.no_grad():
    for data in test_loader:
      data_input = data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)
      target = data.reshape(-1,32,16)[:,:,1:]
      data_input, target = data_input.cuda(),target.cuda()
      np.savetxt('prediction_test',model(data_input).cpu())
      np.savetxt('./test_data',data.reshape(32,-1))
      import pdb
      pdb.set_trace()
  




if __name__ == '__main__':
  model = Conv1d_Model()
  model.to(device="cuda")
  data=np.vstack((generate_snapshots(1,3000,2,512,F1),
  generate_snapshots(1,3000,2,512,F2),
  generate_snapshots(1,4000,2,512,F3))) #10000*512 snapshots
  data = data.reshape(-1,32,16)

  optimizer = optim.Adam(model.parameters(), lr = 1e-2)
  select_train = np.random.choice(range(len(data)), int(0.8*len(data)),replace=False)
  select_test = np.setdiff1d(range(len(data)), select_train)
  train_kwargs = {'batch_size': 32}
  test_kwargs = {'batch_size': 32}

  ### Every time during training, we should first sample several tasks to make the train loader
  ### for test loader, we need to re-sample several other tasks
  train_loader = torch.utils.data.DataLoader(data[select_train], **train_kwargs)
  test_loader = torch.utils.data.DataLoader(data[select_test], **test_kwargs)

  for epoch in range(1, 101):
    train(model, train_loader, optimizer, epoch)
  
  test(model, test_loader)

  # meta_system = ExperimentBuilder(model=model,data=data)
  # meta_system.run_experiment()
