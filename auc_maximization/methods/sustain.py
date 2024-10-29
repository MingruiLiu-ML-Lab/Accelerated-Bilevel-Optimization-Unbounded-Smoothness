from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
import torch
import copy
import numpy as np
from .RNN_net import RNN
from aucloss import AUCMLoss, roc_auc_score
GLOVE_DIM=300
class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.outer_update_lr  = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.inner_model = RNN(
                word_embed_dim=args.word_embed_dim,
                encoder_dim=args.encoder_dim,
                n_enc_layers=args.n_enc_layers,
                dpout_model=0.0,
                dpout_fc=0.0,
                fc_dim=args.fc_dim,
                n_classes=args.n_classes,
                pool_type=args.pool_type,
                linear_fc=args.linear_fc
            )
        self.inner_model_old = RNN(
            word_embed_dim=args.word_embed_dim,
            encoder_dim=args.encoder_dim,
            n_enc_layers=args.n_enc_layers,
            dpout_model=0.0,
            dpout_fc=0.0,
            fc_dim=args.fc_dim,
            n_classes=args.n_classes,
            pool_type=args.pool_type,
            linear_fc=args.linear_fc
        )
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.a_old = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b_old = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.grad_y_old = None
        self.grad_x_old = None
        self.alpha_old = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.upper_variables = [self.a, self.b] + list(self.inner_model.parameters())
        self.upper_variables_old = [self.a, self.b] + list(self.inner_model.parameters())
        self.outer_optimizer = SGD(self.upper_variables, lr=self.outer_update_lr)
        self.inner_optimizer = SGD([self.alpha], lr=self.inner_update_lr)
        self.aucloss = AUCMLoss(self.a, self.b, self.alpha)
        self.aucloss_old = AUCMLoss(self.a_old, self.b_old, self.alpha_old)
        self.inner_model.train()
        self.beta = args.beta
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training=True, epoch=0):
        task_aucs = []
        task_loss = []

        for step, data in enumerate(train_loader):
            self.inner_model.to(self.device)
            self.inner_model_old.to(self.device)

            if step == 0 and epoch==0:
                input_val, label_id_val, data_indx_val = next(iter(val_loader))
                outputs_val = predict(self.inner_model, input_val)
                outer_loss = self.aucloss(outputs_val, label_id_val.to(self.device))
                grad_x = self.stocbio(outer_loss, next(iter(train_loader)), next(iter(train_loader)))
                for i, param in enumerate(self.upper_variables):
                    param.grad = grad_x[i]
                self.inner_optimizer.zero_grad()
                input, label_id, data_indx = next(iter(train_loader))
                outputs = predict(self.inner_model, input)
                inner_loss = -self.aucloss(outputs, label_id.to(self.device))
                grad_y = torch.autograd.grad(inner_loss, self.alpha)
                self.alpha.grad = grad_y[0].detach()
                self.inner_model_old = copy.deepcopy(self.inner_model)
                self.grad_x_old = copy.deepcopy(grad_x)
                self.grad_y_old = copy.deepcopy(grad_y)
                self.a_old.data = self.a.data.clone()
                self.b_old.data = self.b.data.clone()
                self.alpha_old.data = self.alpha.data.clone()
                self.upper_variables_old = [self.a_old, self.b_old] + list(self.inner_model_old.parameters())
            else:
                input_val, label_id_val, data_indx_val = next(iter(val_loader))
                outputs_val = predict(self.inner_model, input_val)
                outer_loss = self.aucloss(outputs_val, label_id_val.to(self.device))
                train_batch = next(iter(train_loader))
                val_batch = next(iter(train_loader))
                grad_x = self.stocbio(outer_loss, train_batch, val_batch)
                outputs_old = predict(self.inner_model_old, input_val)
                outer_loss_old = self.aucloss_old(outputs_old, label_id_val.to(self.device))
                grad_x_on_old_model = self.stocbio_old(outer_loss_old, train_batch, val_batch)

                for  (gx, gxo, gxoo)  in zip( grad_x, self.grad_x_old, grad_x_on_old_model):
                    new_grad = gx.data.detach() + (1 - self.beta) * (gxo.data.detach() - gxoo.data.detach())
                    gxo.data = new_grad

                input, label, data_idx = next(iter(train_loader))
                outputs = predict(self.inner_model, input)
                inner_loss = -self.aucloss(outputs, label.to(self.device))
                grad_y = torch.autograd.grad(inner_loss, self.alpha)

                outputs = predict(self.inner_model_old, input)
                inner_loss = -self.aucloss_old(outputs, label.to(self.device))
                grad_y_on_old_model = torch.autograd.grad(inner_loss, self.alpha_old)

                self.grad_y_old[0].data = grad_y[0].data.detach() + (1-self.beta) * (self.grad_y_old[0].detach() - grad_y_on_old_model[0].detach())
                self.alpha_old.data = self.alpha.data.clone()
                self.alpha.grad = self.grad_y_old[0].data.detach()
                self.a_old.data = self.a.data.clone()
                self.b_old.data = self.b.data.clone()
                self.inner_model_old = copy.deepcopy(self.inner_model)
                self.upper_variables_old = [self.a_old, self.b_old] + list(self.inner_model_old.parameters())
                for p, g in zip(self.upper_variables, self.grad_x_old):
                    p.grad = g.data.clone()

            self.inner_optimizer.step()
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            self.inner_optimizer.zero_grad()

            logits = F.softmax(outputs_val, dim=1)[:, -1]
            label_id = label_id_val.detach().cpu().numpy().tolist()

            auc = roc_auc_score(label_id, logits.detach().cpu().numpy())
            task_aucs.append(auc)
            task_loss.append(outer_loss.detach().cpu().numpy())
            torch.cuda.empty_cache()

            print(f'Task loss: {outer_loss.detach().item():.4f}, Task auc: {auc:.4f}')

        for param_group in self.outer_optimizer.param_groups:
            param_group['lr'] =  self.args.outer_update_lr * ((1 / (epoch + 2)) ** (1 / 3))
            lr = param_group['lr']
            print(f'Outer LR: {lr}')

        for param_group in self.inner_optimizer.param_groups:
            param_group['lr'] =  self.args.inner_update_lr * ((1 / (epoch + 2)) ** (1 / 3))
            lr = param_group['lr']
            print(f'Inner LR: {lr}')

        return np.mean(task_aucs), np.mean(task_loss)

    def collate_pad_(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0])==list else  data_points[1]
        targets = data_points[1] if type(data_points[0])==list else  data_points[0]

        # Get sentences for batch and their lengths.
        s_lens = np.array([sent.shape[0] for sent in s_embeds])
        max_s_len = np.max(s_lens)
        # Encode sentences as glove vectors.
        bs = len(data_points[0])
        s_embed = np.zeros((max_s_len, bs, GLOVE_DIM))
        for i in range(bs):
            e = s_embeds[i]
            if len(e) <= 0:
                s_lens[i] = 1
            s_embed[: len(e), i] = e.copy()
        embeds = torch.from_numpy(s_embed).float().to(self.device)
        targets = torch.LongTensor(targets).to(self.device)
        return (embeds, s_lens), targets

    def test(self, test_loader):

        task_aucs = []
        task_loss = []

        self.inner_model.to(self.device)
        for step, data in enumerate(test_loader):
            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model, q_input)
            q_loss = self.aucloss(q_outputs, q_label_id.to(self.device))

            q_logits = F.softmax(q_outputs, dim=1)[:, -1]
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            auc = roc_auc_score(q_label_id, q_logits.detach().cpu().numpy())
            task_aucs.append(auc)
            task_loss.append(q_loss.detach().cpu().numpy())
            torch.cuda.empty_cache()
            print(f'Task loss: {q_loss.detach().cpu().item():.4f}, Task auc: {auc:.4f}')
        return np.mean(task_aucs), np.mean(task_loss)

    def stocbio(self, loss, train_data_batch, val_data_batch):
        train_data, train_labels, train_indx = train_data_batch
        val_data, val_labels, val_indx = val_data_batch
        Fy_gradient = torch.autograd.grad(loss, self.alpha, retain_graph=True)
        F_gradient = Fy_gradient[0]
        v_0 = F_gradient.detach()
        # Fx_gradient = [g_param.view(-1) for g_param in Fx_gradient]
        Fx_gradient = torch.autograd.grad(loss, self.upper_variables)
        # Hessian
        z_list = []
        outputs = predict(self.inner_model, train_data)
        inner_loss = -self.aucloss(outputs, train_labels.to(self.device))
        G_gradient = []
        Gy_gradient = torch.autograd.grad(inner_loss, self.alpha, create_graph=True)

        for g_grad, param in zip(Gy_gradient, self.alpha):
            G_gradient.append((param - self.args.neumann_lr * g_grad).view(-1))
        # G_gradient = torch.reshape(torch.hstack(G_gradient), [-1])

        for _ in range(self.args.hessian_q):
            # Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(G_gradient, self.alpha, grad_outputs=v_0, retain_graph=True)
            v_0 = v_new[0].data.detach()
            z_list.append(v_0)
        index = np.random.randint(self.args.hessian_q)
        v_Q = self.args.neumann_lr * z_list[index]
        # Gyx_gradient
        outputs = predict(self.inner_model, val_data)
        inner_loss = -self.aucloss(outputs, val_labels.to(self.device))
        Gy_gradient = torch.autograd.grad(inner_loss, self.alpha, retain_graph=True, create_graph=True)
        Gy_params = Gy_gradient[0]
        Gyxv_gradient = torch.autograd.grad(Gy_params, self.upper_variables, grad_outputs= v_Q,  allow_unused=True)
        for i, (f_x, g_yxv) in enumerate(zip(Fx_gradient, Gyxv_gradient)):
             if g_yxv is not None:
                f_x.data -=  g_yxv.data
             else:
                f_x.data -= torch.zeros(1).cuda()
        return Fx_gradient

    def stocbio_old(self, loss, train_data_batch, val_data_batch):
        train_data, train_labels, train_indx = train_data_batch
        val_data, val_labels, val_indx = val_data_batch
        Fy_gradient = torch.autograd.grad(loss, self.alpha_old, retain_graph=True)
        F_gradient = Fy_gradient[0]
        v_0 = F_gradient.detach()
        # Fx_gradient = [g_param.view(-1) for g_param in Fx_gradient]
        Fx_gradient = torch.autograd.grad(loss, self.upper_variables_old)
        # Hessian
        z_list = []
        outputs = predict(self.inner_model_old, train_data)
        inner_loss = -self.aucloss_old(outputs, train_labels.to(self.device))
        G_gradient = []
        Gy_gradient = torch.autograd.grad(inner_loss, self.alpha_old, create_graph=True)

        for g_grad, param in zip(Gy_gradient, self.alpha_old):
            G_gradient.append((param - self.args.neumann_lr * g_grad).view(-1))
        # G_gradient = torch.reshape(torch.hstack(G_gradient), [-1])

        for _ in range(self.args.hessian_q):
            # Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(G_gradient, self.alpha_old, grad_outputs=v_0, retain_graph=True)
            v_0 = v_new[0].data.detach()
            z_list.append(v_0)
        index = np.random.randint(self.args.hessian_q)
        v_Q = self.args.neumann_lr * z_list[index]

        # Gyx_gradient
        outputs = predict(self.inner_model_old, val_data)
        inner_loss = -self.aucloss_old(outputs, val_labels.to(self.device))
        Gy_gradient = torch.autograd.grad(inner_loss, self.alpha_old, retain_graph=True, create_graph=True)
        Gy_params = Gy_gradient[0]
        Gyxv_gradient = torch.autograd.grad(Gy_params, self.upper_variables_old, grad_outputs= v_Q,  allow_unused=True)
        for i, (f_x, g_yxv) in enumerate(zip(Fx_gradient, Gyxv_gradient)):
             if g_yxv is not None:
                f_x.data -=  g_yxv.data
             else:
                f_x.data -= torch.zeros(1).cuda()
        return Fx_gradient


def predict(net, inputs):
    """ Get predictions for a single batch. """
    s_embed, s_lens = inputs
    outputs = net((s_embed.cuda(), s_lens))
    return outputs


