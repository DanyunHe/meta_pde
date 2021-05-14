from nn_architecture import Conv1d_Model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from equations import generate_equations
from copy import deepcopy

# Generate snapshots of PDE solutions for training 
def generate_snapshots(duration=1,num_frame=10000,length=1,n=512,func=F1):
  snapshots = []
  x=np.linspace(0,length,n)
  for t in np.linspace(0.01,duration,num_frame):
    snapshots.append(func(x,t))
  snapshots = np.array(snapshots)
  return snapshots

def get_inner_loop_parameter_dict(params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if "norm_layer" not in name:
                    param_dict[name] = param.cuda()

        return param_dict

# Update inner loop parameters 
def update_inner_params(names_weights_dict, names_grads_wrt_params_dict, inner_loop_lr):
  updated_names_weights_dict = dict()
  for key in names_grads_wrt_params_dict.keys():
    updated_names_weights_dict[key] = names_weights_dict[key] - inner_loop_lr *  names_grads_wrt_params_dict[key]
  return updated_names_weights_dict

# Calculate loss 
def _l2_loss(adapted, prior):
    loss = nn.MSELoss(reduction='mean')
    l2_term = 0.
    for key in adapted.keys():
        l2_term += loss(adapted[key],prior[key])
    return l2_term

# Do inner loop update 
def apply_inner_loop_update(loss, names_weights_copy, names_weights_copy_org, inner_loop_lr):
  loss+=_l2_loss(names_weights_copy, names_weights_copy_org)*0.4
  model.zero_grad(params = names_weights_copy)

  grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=True, allow_unused=True)

  names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
  names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}
  for key, grad in names_grads_copy.items():
      if grad is None:
          print('Grads not found for inner loop parameter', key)
      names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

  names_weights_copy = update_inner_params(names_weights_dict=names_weights_copy,
      names_grads_wrt_params_dict=names_grads_copy,inner_loop_lr = inner_loop_lr)
  
  num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
  names_weights_copy = {
      name.replace('module.', ''): value.unsqueeze(0).repeat(
          [num_devices] + [1 for i in range(len(value.shape))]) for
      name, value in names_weights_copy.items()}

  return names_weights_copy


# Model training 
def train(model, task_train_data, task_test_data, optimizer, epoch, n_inner_step=10):
  
  model.train()


  names_weights_copy = get_inner_loop_parameter_dict(model.named_parameters())

  num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

  names_weights_copy = {
      name.replace('module.', ''): value.unsqueeze(0).repeat(
          [num_devices] + [1 for i in range(len(value.shape))]) for
      name, value in names_weights_copy.items()}

  names_weights_copy_org = get_inner_loop_parameter_dict(model.named_parameters())
  names_weights_copy_org = {
      name.replace('module.', ''): value.unsqueeze(0).repeat(
          [num_devices] + [1 for i in range(len(value.shape))]) for
      name, value in names_weights_copy_org.items()}


  inner_loop_lr=1e-3
  # print(model.state_dict()['conv1d_0'][0])

  task_train_input = torch.tensor(task_train_data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)).cuda() ## shape turns to (10,1,32)
  task_train_target = torch.tensor(task_train_data.reshape(-1,32,16)[:,:,1:]).cuda()

  task_test_input = torch.tensor(task_test_data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)).cuda() ## shape turns to (190,1,32)
  task_test_target = torch.tensor(task_test_data.reshape(-1,32,16)[:,:,1:]).cuda()

  for num_steps in range(n_inner_step):
    output = model.forward(task_train_input, params = names_weights_copy)
    loss = nn.MSELoss(reduction='mean')(output, task_train_target.reshape(-1,480))
    # print("inner loss:",loss)
    names_weights_copy = apply_inner_loop_update(loss,names_weights_copy, names_weights_copy_org, inner_loop_lr=inner_loop_lr)

  task_test_output = model.forward(task_test_input, params = names_weights_copy)
  test_loss = nn.MSELoss(reduction='mean')(task_test_output, task_test_target.reshape(-1,480))

  # print(model.state_dict()['conv1d_0'][0])
  optimizer.zero_grad()
  test_loss.backward()
  for name, param in model.named_parameters():
      if param.requires_grad:
          param.grad.data.clamp_(-10, 10)
  # print("test_loss:",test_loss)
  optimizer.step()
  # print(test_loss)
  return test_loss.detach().cpu().numpy()

# Model testing 
def test(model, task_train_data, task_test_data, n_inner_step=10):
  model.eval()
  
  names_weights_copy = get_inner_loop_parameter_dict(model.named_parameters())

  num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

  names_weights_copy = {
      name.replace('module.', ''): value.unsqueeze(0).repeat(
          [num_devices] + [1 for i in range(len(value.shape))]) for
      name, value in names_weights_copy.items()}

  names_weights_copy_org = get_inner_loop_parameter_dict(model.named_parameters())
  names_weights_copy_org = {
      name.replace('module.', ''): value.unsqueeze(0).repeat(
          [num_devices] + [1 for i in range(len(value.shape))]) for
      name, value in names_weights_copy_org.items()}

  inner_loop_lr=1e-3


  task_train_input = torch.tensor(task_train_data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)).cuda() ## shape turns to (10,1,32)
  task_train_target = torch.tensor(task_train_data.reshape(-1,32,16)[:,:,1:]).cuda()

  task_test_input = torch.tensor(task_test_data.reshape(-1,32,16)[:,:,:1].reshape(-1,1,32)).cuda() ## shape turns to (190,1,32)
  task_test_target = torch.tensor(task_test_data.reshape(-1,32,16)[:,:,1:]).cuda()

  for num_steps in range(n_inner_step):
    output = model.forward(task_train_input, params = names_weights_copy)
    loss = nn.MSELoss(reduction='mean')(output, task_train_target.reshape(-1,480))
    names_weights_copy = apply_inner_loop_update(loss,names_weights_copy, names_weights_copy_org,inner_loop_lr=inner_loop_lr)
  with torch.no_grad():
    task_test_output = model.forward(task_test_input, params = names_weights_copy)
    test_loss = nn.MSELoss(reduction='mean')(task_test_output, task_test_target.reshape(-1,480))

  # with torch.no_grad():
  #   task_test_output = model.forward(task_test_input, weights_copy=weights_copy, bias_copy=bias_copy, weights_linear_copy=weights_linear_copy, bias_linear_copy=bias_linear_copy)
  #   test_loss = nn.MSELoss(reduction='mean')(task_test_output, task_test_target.reshape(-1,480))

  # import pdb
  # pdb.set_trace()
  np.savetxt('prediction_test',task_test_output.cpu())
  np.savetxt('./test_data',task_test_input.cpu().numpy().reshape(-1,32))
  np.savetxt('./test_target',task_test_target.cpu().numpy().reshape(-1,480))
  #import pdb
  #pdb.set_trace()
  # loss = nn.MSELoss(reduction='mean')(prediction, target.reshape(-1,480))
  return test_loss.cpu().numpy()
  
# Generate n different tasks 
def load_all_tasks(n_task):
  data=[]
  for i in range(n_task):
    F=generate_equations(i)
    print(F(1,1))
    data.append(generate_snapshots(func=F))

  return np.array(data)

if __name__ == '__main__':
  """Use random seed to control repeatability"""
  RANDOM_SEED = 0
  n_inner_step = 0
  model = Conv1d_Model()
  model.to(device="cuda")
  
  #### Prepare 20 tasks of different F. Saving data as (10000,512) arrays
  #### Then load all data as (20,10000,512) arrays
  n_task = 45
  inner_loop_data_number = 10
  inner_loop_data_number_test = 5
  data = load_all_tasks(n_task=n_task)
  #### The last two dimensions of data should be (32,16)
  #### eg, data = data.reshape(-1,32,16)

  """Separate to observed tasks (20) for meta-training (task-specific training + task specific testing) and target tasks (5) for meta-testing"""
  observed_tasks = data[:n_task-5,:,:]
  target_tasks = data[n_task-5:,:,:]
  # import pdb
  # pdb.set_trace()
  optimizer = optim.Adam(model.parameters(), lr = 1e-3)
  # select_train = np.random.choice(range(len(data)), int(0.8*len(data)),replace=False)
  # select_test = np.setdiff1d(range(len(data)), select_train)
  train_kwargs = {'batch_size': 32}
  test_kwargs = {'batch_size': 32}

  ### Every time during training, we should first sample several tasks to make the train loader
  ### for test loader, we need to re-sample several other tasks
  # train_loader = torch.utils.data.DataLoader(data[select_train], **train_kwargs)
  # test_loader = torch.utils.data.DataLoader(data[select_test], **test_kwargs)
  """The data for task specific training and task specific testing should be separated
     Choose 10 snapshots for task-specific training and 190 for task-specific testing
  """
  np.random.seed(RANDOM_SEED)
  observed_train_data_idx = [np.random.choice(range(10000),200, replace = False)[:inner_loop_data_number] for i in range(n_task-5)]
  np.random.seed(RANDOM_SEED)
  observed_test_data_idx = [np.random.choice(range(10000),200, replace = False)[inner_loop_data_number:] for i in range(n_task-5)]
  for step in range(1, 5001):
    #### Use step instead of epoch. No well-defined epoch.
    """select a task out of 20 available tasks"""
    sample_observed_task = np.random.choice(range(n_task-5)) 
    task_train_data = observed_tasks[sample_observed_task][observed_train_data_idx[sample_observed_task]]
    task_test_data = observed_tasks[sample_observed_task][observed_test_data_idx[sample_observed_task]]
    

    loss=train(model, task_train_data, task_test_data, optimizer, step, n_inner_step)
    # print(loss)
    if step%100==0:
        sample_target_task = np.random.choice(range(5)) # select a task out of 20 available tasks


        np.random.seed(RANDOM_SEED)
        target_train_data_idx = np.random.choice(range(10000),200, replace = False)[:inner_loop_data_number_test]
        np.random.seed(RANDOM_SEED)
        target_test_data_idx = np.random.choice(range(10000),200, replace = False)[inner_loop_data_number_test:]

        task_train_data = target_tasks[sample_target_task][target_train_data_idx]
        task_test_data = target_tasks[sample_target_task][target_test_data_idx]

        test_loss=test(model, task_train_data, task_test_data, n_inner_step=2)
        print("step: ",step,"train loss ",loss,"test loss: ",test_loss)


  # meta_system = ExperimentBuilder(model=model,data=data)
  # meta_system.run_experiment()
