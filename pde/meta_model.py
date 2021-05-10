import torch
import torch.nn as nn
from nn_architecture import Conv1d_Model

class MetaLearner(nn.Module):
    def __init__:
        self.predictor = Conv1d_Model

        self.self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.loss=nn.MSELoss()

        ## Overall copy of weights and biases
    
    def run_train_iter(self, data_batch, epoch):
        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        loss, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch)

        self.meta_update(loss)
        self.optimizer.zero_grad()

    def run_validation_iter():

    def run_test_iter():


    def meta_update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def apply_inner_loop_update():
    
    def forward(num_inner_steps):
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            ## TODO: Copy weights and biases for inner loop
            
            for num_step in range(num_inner_steps):
                support_loss, _ = self.net_forward(x=x_support_set_task,
                                                               y=y_support_set_task,
                                                               weights_copy=weights_copy,
                                                               bias_copy=bias_copy)

                ### perform inner loop update
                final_loss+=support_loss
                support_loss.backward()
                self.optimizer.step()
                ## update weights_copy, bias_copy


            ### Collect final loss for meta update
            meta_update(final_loss)

    def net_forward(x, y, weights, bias):
        alpha = self.predictor.forward(x=x,weights_copy=weights, bias_copy = bias)

        ##rollout(alpha)

        preds = alpha@x
        loss = loss(preds,y)
        return losss, preds



