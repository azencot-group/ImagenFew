import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import kl_loss
from .neuralCDE import NeuralCDE
from .modules import FinalTanh


def reparameterize(mean, logvar, random_sampling=True):
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean


class VKEncoderIrregular(nn.Module):
    def __init__(self, args):
        super(VKEncoderIrregular, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm
        self.num_layers = self.args.num_layers

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        ode_func = FinalTanh(self.inp_dim, self.hidden_dim, self.hidden_dim, self.num_layers)
        self.emb = NeuralCDE(func=ode_func, input_channels=self.inp_dim,
                    hidden_channels=self.hidden_dim, output_channels=self.hidden_dim).to(args.device)
        self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bidirectional=True,
                           num_layers=1, batch_first=True)


    def forward(self, time, train_coeffs, final_index):
        # encode
        h = self.emb(time, train_coeffs, final_index)
        h, _ = self.rnn(h)
        if self.batch_norm:
            h = self.b_norm(torch.permute(h, (0, 2, 1)))
            h = torch.permute(h, (0, 2, 1))  # permute back to b x s x c
        return h


class VKEncoder(nn.Module):
    def __init__(self, args, num_layers=3):
        super(VKEncoder, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.rnn = nn.GRU(input_size=self.inp_dim, hidden_size=self.hidden_dim, bidirectional=True,
                           num_layers=args.num_layers, batch_first=True)

    def forward(self, x):
        # encode
        h, _ = self.rnn(x)  # b x seq_len x channels
        if self.batch_norm:
            h = self.b_norm(torch.permute(h, (0, 2, 1)))
            h = torch.permute(h, (0, 2, 1))  # permute back to b x s x c
        return h


class VKDecoder(nn.Module):
    def __init__(self, args):
        super(VKDecoder, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim

        self.rnn = nn.GRU(input_size=self.z_dim, hidden_size=self.hidden_dim, bidirectional=True,
                           num_layers=args.num_layers, batch_first=True)

        self.linear = nn.Linear(self.args.hidden_dim * 2, self.args.inp_dim)


    def forward(self, z):
        # decode
        h, _ = self.rnn(z)
        x_hat = nn.functional.sigmoid(self.linear(h))
        return x_hat


class KoVAE(nn.Module):
    def __init__(self, args):
        super(KoVAE, self).__init__()
        self.args = args
        self.z_dim = args.z_dim  # latent
        self.channels = args.inp_dim  # seq channel (multivariate features)
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.seq_len = args.seq_len
        self.pinv_solver = args.pinv_solver
        self.missing_value = args.missing_value

        if self.missing_value > 0.:
            self.encoder = VKEncoderIrregular(self.args)
        else:
            self.encoder = VKEncoder(self.args)
        self.decoder = VKDecoder(self.args)

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.z_prior_gru = nn.GRUCell(self.z_dim, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # ----- Posterior of sequence
        self.z_mean = nn.Linear(self.hidden_dim * 2, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim * 2, self.z_dim)

        self.names = ['total', 'rec', 'kl', 'pred_prior']


    def forward(self, x, time=None, final_index=None):

        # ------------- ENCODING PART -------------
        if time is not None and final_index is not None:
            z = self.encoder(time, x, final_index)
        else:
            z = self.encoder(x)

        # variational part for input
        z_mean = self.z_mean(z)
        z_logvar = self.z_logvar(z)
        z_post = reparameterize(z_mean, z_logvar, random_sampling=True)

        Z_enc = {'mean': z_mean, 'logvar': z_logvar, 'sample': z_post}

        # # ------------- PRIOR PART -------------
        z_mean_prior, z_logvar_prior, z_out = self.sample_prior(z.size(0), self.seq_len, random_sampling=True)
        Z_enc_prior = {'mean': z_mean_prior, 'logvar': z_logvar_prior, 'sample': z_out}

        # pass z_post instead of z_pred
        x_rec = self.decoder(z_post)

        return x_rec, Z_enc, Z_enc_prior

    def compute_operator_and_pred(self, z):
        z_past, z_future = z[:, :-1], z[:, 1:]  # split latent

        # solve linear system (broadcast)
        if self.pinv_solver:
            Ct = torch.linalg.pinv(z_past.reshape(-1, self.z_dim)) @ z_future.reshape(-1, self.z_dim)

        else:
            # self.qr_solver
            Q, R = torch.linalg.qr(z_past.reshape(-1, self.z_dim))
            B = Q.T @ z_future.reshape(-1, self.z_dim)
            Ct = torch.linalg.solve_triangular(R, B, upper=True)

        # predict (broadcast)
        z_pred = z_past @ Ct

        err = .0
        z_hat = z_past
        for jj in range(self.args.num_steps):
            z_hat = z_hat @ Ct
            err += (F.mse_loss(z_hat[:, :-jj or None], z[:, (jj + 1):]) / torch.norm(z_hat[:, :-jj or None], p='fro'))

        return Ct, z_pred, err

    def loss(self, x, x_rec, Z_enc, Z_enc_prior):
        """
        :param x: The original sequence input
        :param x_rec: The reconstructed sequence
        :param Z_enc: Dictionary of posterior modeling {mean, logvar and sample}
        :param Z_enc_prior: Dictionary of prior modeling {mean, logvar and sample}
        :return: loss value
        """

        # PENALTIES
        a0 = self.args.w_rec
        a1 = self.args.w_kl
        a2 = self.args.w_pred_prior

        batch_size = x.size(0)

        z_post_mean, z_post_logvar, z_post = Z_enc['mean'], Z_enc['logvar'], Z_enc['sample']
        z_prior_mean, z_prior_logvar, z_prior = Z_enc_prior['mean'], Z_enc_prior['logvar'], Z_enc_prior['sample']
        ## Ap after sampling ##
        Ct_prior, z_pred_prior, pred_err_prior = self.compute_operator_and_pred(z_prior)

        loss = .0
        if self.args.w_rec:
            recon = F.mse_loss(x_rec, x, reduction='sum') / batch_size
            loss = a0 * recon
            agg_losses = [loss]

        if self.args.w_kl:
            kld_z = kl_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
            # kld_z = torch.clamp(kld_z, min=self.budget)
            kld_z = torch.sum(kld_z) / batch_size
            loss += a1 * kld_z
            agg_losses.append(kld_z)

        if self.args.w_pred_prior:
            loss += a2 * pred_err_prior
        agg_losses.append(pred_err_prior)

        agg_losses = [loss] + agg_losses
        return tuple(agg_losses)


    def sample_data(self, n_sample):
        # sample from prior
        z_mean_prior, z_logvar_prior, z_out = self.sample_prior(n_sample, self.seq_len, random_sampling=True)
        x_rec = self.decoder(z_out)
        return x_rec

    # ------ sample z purely from learned LSTM prior with arbitrary seq ------
    def sample_prior(self, n_sample, seq_len, random_sampling=True):

        batch_size = n_sample

        # z_out = None  # This will ultimately store all z_s in the format [batch_size, seq_len, z_dim]
        z_logvars, z_means, z_out = self.zeros_init(batch_size, seq_len)

        # initialize arbitrary input (zeros) and hidden states.
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(seq_len):
            h_t = self.z_prior_gru(z_t, h_t)

            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = reparameterize(z_mean_t, z_logvar_t, random_sampling)

            z_out[:, i] = z_t
            z_means[:, i] = z_mean_t
            z_logvars[:, i] = z_logvar_t

        return z_means, z_logvars, z_out

    def zeros_init(self, batch_size, seq_len):
        z_out = torch.zeros(batch_size, seq_len, self.z_dim).cuda()
        z_means = torch.zeros(batch_size, seq_len, self.z_dim).cuda()
        z_logvars = torch.zeros(batch_size, seq_len, self.z_dim).cuda()
        return z_logvars, z_means, z_out