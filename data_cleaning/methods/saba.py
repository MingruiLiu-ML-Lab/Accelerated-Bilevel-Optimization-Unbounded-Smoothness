import copy

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from .RNN_net import RNN, NLIRNN

GLOVE_DIM = 300

def saga_update(grad, last_grad, hist_grad_list, hist_length=10):
    if len(hist_grad_list) < hist_length:
        hist_grad_list.append(grad)
    else:
        replace_index = np.random.randint(hist_length)
        hist_grad_list[replace_index] = grad
    if not isinstance(grad, tuple):
        return_vector = torch.zeros_like(grad, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in range(len(hist_grad_list)):
            return_vector += hist_grad_list[i]
        return_vector /= len(hist_grad_list)
        return_vector += grad.detach() - last_grad if last_grad != None else grad
    else:
        return_vector = [torch.zeros_like(x, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for x in grad]
        for j in range(len(return_vector)):
            for i in range(len(hist_grad_list)):
                return_vector[j] += hist_grad_list[i][j]
            return_vector[j] /= len(hist_grad_list)
        for j in range(len(return_vector)):
            if last_grad != None:
                return_vector[j] += grad[j].detach() - last_grad[j]
            else:
                return_vector[j] += grad[j].detach()
    return return_vector, hist_grad_list

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, training_size):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collate_pad_ = self.collate_pad if args.data=='news_data' else self.collate_pad_snli
        self.training_size = training_size

        if args.data == 'snli':
            self.inner_model = NLIRNN(
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

        self.lambda_x =  torch.ones((self.training_size)).to(self.device)
        self.lambda_x.requires_grad=True
        self.last_G_grad_y = None
        self.hist_G_grad_y = []
        self.last_F_grad_y = None
        self.hist_F_grad_y = []
        self.last_jvp = None
        self.hist_jvp = []
        self.last_hvp = None
        self.hist_hvp = []
        self.z_params = [torch.zeros_like(x).to(self.device) for x in self.inner_model.parameters()]
        self.outer_optimizer = SGD([self.lambda_x], lr=self.outer_update_lr)
        self.inner_optimizer = SGD(self.inner_model.parameters(), lr=self.inner_update_lr)
        self.inner_model.train()
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training=True, epoch=0):
        task_accs = []
        task_loss = []
        self.inner_model.to(self.device)
        for step, data in enumerate(train_loader):
            all_loss = []
            input, label_id, data_indx = data
            outputs = predict(self.inner_model, input)
            inner_loss = torch.mean(torch.sigmoid(self.lambda_x[data_indx])*self.criterion(outputs, label_id.to(self.device))) + 0.0001 * sum(
                [x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()
            g_grad = torch.autograd.grad(inner_loss, self.inner_model.parameters(), retain_graph=True, create_graph=True)
            G_grad_y, self.hist_G_grad_y = saga_update(g_grad, self.last_G_grad_y, self.hist_G_grad_y)
            self.last_G_grad_y = g_grad
            all_loss.append(inner_loss.item())

            hvp = torch.autograd.grad(g_grad, self.inner_model.parameters(), grad_outputs=self.z_params)
            hvp_update, self.hist_hvp = saga_update(hvp, self.last_hvp, self.hist_hvp)
            self.last_hvp = copy.deepcopy(hvp)
            self.inner_optimizer.zero_grad()
            for param, pg in zip(self.inner_model.parameters(), G_grad_y):
                param.grad = pg
            self.inner_optimizer.step()
            self.inner_optimizer.zero_grad()

            q_input, q_label_id, q_data_indx = next(iter(val_loader))
            q_outputs = predict(self.inner_model, q_input)
            q_loss = torch.mean(self.criterion(q_outputs, q_label_id.to(self.device)))
            hyper_grad = self.hypergradient(self.args, hvp_update, q_loss, next(iter(train_loader)))
            self.lambda_x.grad = hyper_grad[0]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            self.outer_optimizer.zero_grad()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')
        return np.mean(task_accs),  np.mean(task_loss)

    def test(self, test_loader):
        task_accs = []
        task_loss = []

        self.inner_model.to(self.device)
        for step, data in enumerate(test_loader):
            q_input, q_label_id, q_data_indx = data
            q_outputs = predict(self.inner_model, q_input)
            q_loss = torch.mean(self.criterion(q_outputs, q_label_id.to(self.device)))

            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'Task loss: {np.mean(task_loss):.4f}, Task acc: {np.mean(task_accs):.4f}')
        return np.mean(task_accs), np.mean(task_loss)

    def hypergradient(self, args, hvp, loss, query_batch):
        val_data, val_labels, data_idx = query_batch
        Fy_gradient = torch.autograd.grad(loss, self.inner_model.parameters())
        F_grad_y, self.hist_F_grad_y = saga_update(Fy_gradient, self.last_F_grad_y, self.hist_F_grad_y)
        self.last_F_grad_y = Fy_gradient
        for i in range(len(self.z_params)):
            self.z_params[i] -= args.nu * (hvp[i].detach() - F_grad_y[i].detach())
        output = predict(self.inner_model, val_data)
        loss = torch.mean(
            torch.sigmoid(self.lambda_x[data_idx]) * F.cross_entropy(output, val_labels.cuda(), reduction='none')) + 0.0001 * sum(
            [x.norm().pow(2) for x in self.inner_model.parameters()]).sqrt()
        Gy_gradient = torch.autograd.grad(loss, self.inner_model.parameters(), retain_graph=True, create_graph=True)
        Gyxz_gradient = torch.autograd.grad(Gy_gradient, self.lambda_x, grad_outputs=self.z_params)
        Gyxz_gradient_update, self.hist_jvp = saga_update(Gyxz_gradient, self.last_jvp, self.hist_jvp)
        self.last_jvp = Gyxz_gradient
        # F_x = 0 here
        return  Gyxz_gradient_update


    def collate_pad(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple  else data_points[0]
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

    def collate_pad_snli(self, data_points):
        """ Pad data points with zeros to fit length of longest data point in batch. """
        s_embeds = data_points[0] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[1]
        targets = data_points[1] if type(data_points[0]) == list or type(data_points[0]) == tuple else data_points[0]
        s1_embeds = [x for x in s_embeds[0]]
        s2_embeds = [x for x in s_embeds[1]]

        # Get sentences for batch and their lengths.
        s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
        max_s1_len = np.max(s1_lens)
        s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
        max_s2_len = np.max(s2_lens)
        lens = (s1_lens, s2_lens)

        # Encode sentences as glove vectors.
        bs = len(targets)
        s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
        s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
        for i in range(bs):
            e1 = s1_embeds[i]
            e2 = s2_embeds[i]
            s1_embed[: len(e1), i] = e1.copy()
            s2_embed[: len(e2), i] = e2.copy()
            if len(e1) <= 0:
                s1_lens[i] = 1
            if len(e2) <= 0:
                s2_lens[i] = 1
        embeds = (
            torch.from_numpy(s1_embed).float().to(self.device), torch.from_numpy(s2_embed).float().to(self.device)
        )

        # Convert targets to tensor.
        targets = torch.LongTensor(targets).to(self.device)

        return (embeds, lens), targets

def predict(net, inputs):
    """ Get predictions for a single batch. """
    # snli dataset
    (s1_embed, s2_embed), (s1_lens, s2_lens) = inputs
    outputs = net((s1_embed.cuda(), s1_lens), (s2_embed.cuda(), s2_lens))
    return outputs



