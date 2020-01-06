import math
import time

import torch
from torch import nn

from models.utils import from_numpy, to_numpy, make_data_loader
from models.utils import students_t_nll, gaussian_nll, standard_gaussian_nll, gaussian_kl_divergence, reparameterize


class StudentsTNetwork(nn.Module):
    def __init__(self, n_in, n_latent, n_h):
        super(StudentsTNetwork, self).__init__()

        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h

        # Encoder
        self.le1 = nn.Sequential(
            nn.Linear(n_in, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        )
        self.le2_mu = nn.Linear(n_h, n_latent)
        self.le2_ln_var = nn.Linear(n_h, n_latent)

        # Decoder
        self.ld1 = nn.Sequential(
            nn.Linear(n_latent, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        )
        self.ld2_ln_df = nn.Linear(n_h, n_in)
        self.ld2_loc = nn.Linear(n_h, n_in)
        self.ld2_ln_scale = nn.Linear(n_h, n_in)

    def encode(self, x):
        h = self.le1(x)
        return self.le2_mu(h), self.le2_ln_var(h)

    def decode(self, z):
        h = self.ld1(z)
        return self.ld2_ln_df(h), self.ld2_loc(h), self.ld2_ln_scale(h)

    def forward(self, x, k=1):
        # Compute Negative ELBO
        mu_enc, ln_var_enc = self.encode(x)

        RE = 0
        for i in range(k):
            z = reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            ln_df, loc, ln_scale = self.decode(z)
            RE += students_t_nll(x, ln_df=ln_df, loc=loc, ln_scale=ln_scale) / k

        KL = gaussian_kl_divergence(mu=mu_enc, ln_var=ln_var_enc)
        return RE, KL

    def evidence_lower_bound(self, x, k=1):
        RE, KL = self.forward(x, k=k)
        return -1 * (RE + KL)

    def importance_sampling(self, x, k=1):
        mu_enc, ln_var_enc = self.encode(x)
        lls = []
        for i in range(k):
            z = reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            ln_df, loc, ln_scale = self.decode(z)
            ll = -1 * students_t_nll(x, ln_df=ln_df, loc=loc, ln_scale=ln_scale, dim=1)
            ll -= standard_gaussian_nll(z, dim=1)
            ll += gaussian_nll(z, mu=mu_enc, ln_var=ln_var_enc, dim=1)
            lls.append(ll[:, None])

        return torch.cat(lls, dim=1).logsumexp(dim=1) - math.log(k)


class StudentsTVAE:
    def __init__(self, n_in, n_latent, n_h):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = StudentsTNetwork(n_in, n_latent, n_h).to(self.device)

        self.train_losses = []
        self.train_times = []
        self.reconstruction_errors = []
        self.kl_divergences = []
        self.valid_losses = []
        self.min_valid_loss = float("inf")

    def _loss_function(self, x, k=1, beta=1):
        RE, KL = self.network(x, k=k)
        RE_sum = RE.sum()
        KL_sum = KL.sum()
        loss = RE_sum + beta * KL_sum
        return loss, RE_sum, KL_sum

    def fit(self, X, k=1, batch_size=100, learning_rate=0.001, n_epoch=500,
            warm_up=False, warm_up_epoch=100,
            is_stoppable=False, X_valid=None, path=None):

        self.network.train()
        N = X.shape[0]
        data_loader = make_data_loader(X, device=self.device, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        if is_stoppable:
            X_valid = from_numpy(X_valid, self.device)

        for epoch in range(n_epoch):
            start = time.time()

            # warm-up
            beta = 1 * epoch / warm_up_epoch if warm_up and epoch <= warm_up_epoch else 1

            mean_loss = 0
            mean_RE = 0
            mean_KL = 0
            for _, batch in enumerate(data_loader):
                optimizer.zero_grad()
                loss, RE, KL = self._loss_function(batch[0], k=k, beta=beta)
                loss.backward()
                mean_loss += loss.item() / N
                mean_RE += RE.item() / N
                mean_KL += KL.item() / N
                optimizer.step()

            end = time.time()
            self.train_losses.append(mean_loss)
            self.train_times.append(end - start)
            self.reconstruction_errors.append(mean_RE)
            self.kl_divergences.append(mean_KL)

            print(f"epoch: {epoch} / Train: {mean_loss:0.3f} / RE: {mean_RE:0.3f} / KL: {mean_KL:0.3f}", end='')

            if warm_up and epoch < warm_up_epoch:
                print(" / Warm-up", end='')
            elif is_stoppable:
                valid_loss, _, _ = self._loss_function(X_valid, k=k, beta=1)
                valid_loss = valid_loss.item() / X_valid.shape[0]
                print(f" / Valid: {valid_loss:0.3f}", end='')
                self.valid_losses.append(valid_loss)
                self._early_stopping(valid_loss, path)

            print('')

        if is_stoppable:
            self.network.load_state_dict(torch.load(path))

        self.network.eval()

    def _early_stopping(self, valid_loss, path):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            torch.save(self.network.state_dict(), path)
            print(" / Save", end='')

    def encode(self, X):
        mu, ln_var = self.network.encode(from_numpy(X, self.device))
        return to_numpy(mu, self.device), to_numpy(ln_var, self.device)

    def decode(self, Z):
        ln_df, loc, ln_scale = self.network.decode(from_numpy(Z, self.device))
        return to_numpy(ln_df, self.device), to_numpy(loc, self.device), to_numpy(ln_scale, self.device)

    def evidence_lower_bound(self, X, k=1):
        return to_numpy(self.network.evidence_lower_bound(from_numpy(X, self.device), k=k), self.device)

    def importance_sampling(self, X, k=1):
        return to_numpy(self.network.importance_sampling(from_numpy(X, self.device), k=k), self.device)
