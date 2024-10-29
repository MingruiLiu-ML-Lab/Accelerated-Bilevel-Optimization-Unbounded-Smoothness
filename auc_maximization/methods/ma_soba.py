from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
import torch
import numpy as np
from .RNN_net import RNN
from aucloss import AUCMLoss, roc_auc_score
GLOVE_DIM = 300

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
        self.num_labels = args.num_labels
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collate_pad_ = self.collate_pad if args.data=='news_data' else self.collate_pad_snli

        if args.data == 'news_data':
            self.meta_model = RNN(
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
        if args.data == 'snli' or 'sentment140':
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
        param_count = 0
        for param in self.inner_model.parameters():
            param_count += param.numel()
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.upper_variables= [self.a, self.b]+list(self.inner_model.parameters())
        self.hyper_momentum = [torch.zeros(param.size()).to(self.device) for param in
                                       self.upper_variables]
        self.z_params = torch.randn(1, 1)
        self.z_params = nn.init.xavier_uniform_(self.z_params).to(self.device)[0]
        self.outer_optimizer = SGD(self.upper_variables, lr=self.outer_update_lr)
        self.inner_optimizer = SGD([self.alpha], lr=self.inner_update_lr)
        self.aucloss = AUCMLoss(self.a, self.b, self.alpha)
        self.inner_model.train()
        self.beta = args.beta
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, train_loader, val_loader, training=True, epoch=0):
        task_accs = []
        task_loss = []
        self.inner_model.to(self.device)
        for step, data in enumerate(train_loader):
            all_loss = []
            input, label_id, data_indx = data
            outputs = predict(self.inner_model, input)
            inner_loss = -self.aucloss(outputs, label_id.to(self.device))
            inner_loss.backward(retain_graph=True, create_graph=True)
            all_loss.append(inner_loss.item())
            g_grad = self.alpha.grad

            jacob = torch.autograd.grad(g_grad, self.alpha, grad_outputs=self.z_params)

            self.inner_optimizer.step()
            self.inner_optimizer.zero_grad()
            q_input, q_label_id, q_data_indx = next(iter(val_loader))
            q_outputs = predict(self.inner_model, q_input)
            q_loss = self.aucloss(q_outputs, q_label_id.to(self.device))
            valid_batch = next(iter(train_loader))
            self.hypergradient(self.args, jacob[0], q_loss, valid_batch)
            for i, param in enumerate(self.upper_variables):
                param.grad = self.hyper_momentum[i]


            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            q_logits = F.softmax(q_outputs, dim=1)[:, -1]
            q_label_id = q_label_id.detach().cpu().numpy().tolist()
            self.outer_optimizer.zero_grad()
            auc = roc_auc_score(q_label_id, q_logits.detach().cpu().numpy())
            task_accs.append(auc)
            task_loss.append(q_loss.detach().cpu().numpy())
            torch.cuda.empty_cache()
            print(f'Task loss: {q_loss.detach().cpu().item():.4f}, Task auc: {auc:.4f}')
        return np.mean(task_accs),  np.mean(task_loss)

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

    def hypergradient(self, args, jacob_flat, loss, query_batch):
        val_data, val_labels, data_idx = query_batch
        Fy_gradient = torch.autograd.grad(loss, self.alpha, retain_graph=True)
        Fx_gradient = torch.autograd.grad(loss, self.upper_variables)
        self.z_params -= args.nu * (jacob_flat.detach() - Fy_gradient[0].data)

        output = predict(self.inner_model, val_data)
        loss = -self.aucloss(output, val_labels.to(self.device))
        Gy_gradient = torch.autograd.grad(loss, self.alpha, retain_graph=True, create_graph=True)
        Gyxz_gradient = torch.autograd.grad(Gy_gradient, self.upper_variables, grad_outputs=self.z_params.detach(),
                                            allow_unused=True)
        self.hyper_momentum = [
            args.beta * h + (1 - args.beta) * (fx.detach() - Gyxz.detach() if Gyxz is not None else fx.detach()) for
            (h, fx, Gyxz) in
            zip(self.hyper_momentum, Fx_gradient, Gyxz_gradient)]


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
    s_embed, s_lens = inputs
    outputs = net((s_embed.cuda(), s_lens))
    return outputs



