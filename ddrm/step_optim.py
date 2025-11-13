import torch
import math
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


def interpolate_fn(x, xp, yp):
    # Differentiable piecewise linear interpolation.
    # Finds y for a given x, based on keypoints (xp, yp).
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


class NoiseScheduleVP:
    # Defines the noise schedule (VP-SDE) based on DDPM conventions.
    # Provides functions for alpha_t, sigma_t, and lambda_t (half-logSNR).
    # Supports both discrete schedules (from pre-trained alphas_cumprod)
    # and continuous schedules (linear, cosine).
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
    ):

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule.")

        self.schedule = schedule
        if schedule == 'discrete':
            # Handle discrete schedules (e.g., from pre-trained DDPMs)
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                # DDPM's alphas_cumprod is alpha_hat_n, so log(alpha_t_n) = 0.5 * log(alpha_hat_n)
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            # Create keypoints for interpolation
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            # Handle continuous schedules
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        # Computes log(alpha_t) for a given t.
        if self.schedule == 'discrete':
            # Use linear interpolation for discrete schedules
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),
                                  self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        # Computes alpha_t
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        # Computes sigma_t
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        # Computes lambda_t = log(alpha_t / sigma_t) (half-logSNR)
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        # Computes t from a given lambda_t (inverse of marginal_lambda)
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            # Use interpolation in reverse
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                               torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t

    # --- EDM Helper Functions ---
    def edm_sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def edm_inverse_sigma(self, edmsigma):
        alpha = 1 / (edmsigma ** 2 + 1).sqrt()
        sigma = alpha * edmsigma
        lambda_t = torch.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t


class StepOptim(object):
    # Implements the StepOptim algorithm from arXiv:2402.17376.
    # Finds an optimal non-uniform timestep schedule by minimizing
    # a derived global discretization error bound using scipy.optimize.
    def __init__(self, ns):
        super().__init__()
        self.ns = ns  # NoiseScheduleVP object
        self.T = 1.0  # Default end time for VP models

    # --- NumPy wrappers for noise schedule functions ---
    def alpha(self, t):
        t = torch.as_tensor(t, dtype=torch.float64)
        return self.ns.marginal_alpha(t).numpy()

    def sigma(self, t):
        return np.sqrt(1 - self.alpha(t) * self.alpha(t))

    def lambda_func(self, t):
        return np.log(self.alpha(t) / self.sigma(t))

    # --- Helper functions for high-order solver weights (Taylor expansion terms) ---
    def H0(self, h):
        return np.exp(h) - 1

    def H1(self, h):
        return np.exp(h) * h - self.H0(h)

    def H2(self, h):
        return np.exp(h) * h * h - 2 * self.H1(h)

    def H3(self, h):
        return np.exp(h) * h * h * h - 3 * self.H2(h)

    # --- More NumPy wrappers ---
    def inverse_lambda(self, lamb):
        lamb = torch.as_tensor(lamb, dtype=torch.float64)
        return self.ns.inverse_lambda(lamb)

    def edm_sigma(self, t):
        return np.sqrt(1. / (self.alpha(t) * self.alpha(t)) - 1)

    def edm_inverse_sigma(self, edm_sigma):
        alpha = 1 / (edm_sigma * edm_sigma + 1).sqrt()
        sigma = alpha * edm_sigma
        lambda_t = np.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t

    def sel_lambdas_lof_obj(self, lambda_vec, eps):
        # This is the core objective function (Eq. 4.10 in the paper) to be minimized.
        # It calculates the global error bound based on the solver weights (J_n_kp)
        # and the assumed model error (data_err_vec).

        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()
        # lambda_vec holds the intermediate steps; add boundaries T and eps.
        lambda_vec_ext = np.concatenate((np.array([lambda_T]), lambda_vec, np.array([lambda_eps])))
        N = len(lambda_vec_ext) - 1

        hv = np.zeros(N)  # h_i = lambda_i+1 - lambda_i
        for i in range(N):
            hv[i] = lambda_vec_ext[i + 1] - lambda_vec_ext[i]

        elv = np.exp(lambda_vec_ext)
        emlv_sq = np.exp(-2 * lambda_vec_ext)
        alpha_vec = 1. / np.sqrt(1 + emlv_sq)
        sigma_vec = 1. / np.sqrt(1 + np.exp(2 * lambda_vec_ext))

        # data_err_vec defines the assumed model error (epsilon_tilde_t_i in the paper)
        # p=2 (sigma^2 / alpha) is used here.
        data_err_vec = (sigma_vec ** 2) / alpha_vec

        # truncNum stabilizes optimization for low NFEs (per the original paper)
        truncNum = 3
        res = 0.
        c_vec = np.zeros(N)

        # Loop over steps (s) to calculate solver weights (J) and sum the weighted errors
        for s in range(N):
            if s in [0, N - 1]:
                # 1st order (start and end)
                n, kp = s, 1
                J_n_kp_0 = elv[n + 1] - elv[n]
                res += abs(J_n_kp_0 * data_err_vec[n])
            elif s in [1, N - 2]:
                # 2nd order
                n, kp = s - 1, 2
                J_n_kp_0 = -elv[n + 1] * self.H1(hv[n + 1]) / hv[n]
                J_n_kp_1 = elv[n + 1] * (self.H1(hv[n + 1]) + hv[n] * self.H0(hv[n + 1])) / hv[n]
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n + 1] += data_err_vec[n + 1] * J_n_kp_1
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0) ** 2 + (data_err_vec[n + 1] * J_n_kp_1) ** 2)
            else:
                # 3rd order
                n, kp = s - 2, 3
                J_n_kp_0 = elv[n + 2] * (self.H2(hv[n + 2]) + hv[n + 1] * self.H1(hv[n + 2])) / (
                        hv[n] * (hv[n] + hv[n + 1]))
                J_n_kp_1 = -elv[n + 2] * (self.H2(hv[n + 2]) + (hv[n] + hv[n + 1]) * self.H1(hv[n + 2])) / (
                        hv[n] * hv[n + 1])
                J_n_kp_2 = elv[n + 2] * (
                        self.H2(hv[n + 2]) + (2 * hv[n + 1] + hv[n]) * self.H1(hv[n + 2]) + hv[n + 1] * (
                        hv[n] + hv[n + 1]) * self.H0(hv[n + 2])) / (hv[n + 1] * (hv[n] + hv[n + 1]))
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n + 1] += data_err_vec[n + 1] * J_n_kp_1
                    c_vec[n + 2] += data_err_vec[n + 2] * J_n_kp_2
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0) ** 2 + (data_err_vec[n + 1] * J_n_kp_1) ** 2 + (
                            data_err_vec[n + 2] * J_n_kp_2) ** 2)
        res += sum(abs(c_vec))
        return res

    def get_ts_lambdas(self, N, eps, initType):
        # This is the main function called by experiment scripts.
        # It sets up and runs the scipy.optimize.minimize optimizer.
        # N: Number of function evaluations (NFE) desired.
        # eps: The starting time t_0 (e.g., 1e-3).
        # initType: The initialization strategy ('unif_t', 'unif', 'edm', 'quad').

        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()

        # Define the constraints for the optimizer: lambda_T > lambda_1 > ... > lambda_N-1 > lambda_eps
        constr_mat = np.zeros((N, N - 1))
        for i in range(N - 1):
            constr_mat[i][i] = 1.
            constr_mat[i + 1][i] = -1
        lb_vec = np.zeros(N)
        lb_vec[0], lb_vec[-1] = lambda_T, -lambda_eps

        ub_vec = np.zeros(N)
        for i in range(N):
            ub_vec[i] = np.inf
        linear_constraint = LinearConstraint(constr_mat, lb_vec, ub_vec)

        # Create the initial vector (a heuristic schedule) based on initType
        if initType in ['unif', 'unif_origin']:
            lambda_vec_ext = torch.linspace(lambda_T, lambda_eps, N + 1)
        elif initType in ['unif_t', 'unif_t_origin']:
            t_vec = torch.linspace(self.T, eps, N + 1)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['edm', 'edm_origin']:
            rho = 7
            edm_sigma_min, edm_sigma_max = self.edm_sigma(eps).item(), self.edm_sigma(self.T).item()
            edm_sigma_vec = torch.linspace(edm_sigma_max ** (1. / rho), edm_sigma_min ** (1. / rho), N + 1).pow(rho)
            t_vec = self.edm_inverse_sigma(edm_sigma_vec)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['quad', 'quad_origin']:
            t_order = 2
            t_vec = torch.linspace(self.T ** (1. / t_order), eps ** (1. / t_order), N + 1).pow(t_order)
            lambda_vec_ext = self.lambda_func(t_vec)
        else:
            print('InitType not found!')
            return

        if initType in ['unif_origin', 'unif_t_origin', 'edm_origin', 'quad_origin']:
            # '_origin' types are for debugging/baseline, just return the heuristic schedule
            lambda_res = lambda_vec_ext
            t_res = self.inverse_lambda(lambda_res).detach().clone()
        else:
            # Run the optimization
            lambda_vec_init = np.array(lambda_vec_ext[1:-1])
            res = minimize(self.sel_lambdas_lof_obj, lambda_vec_init, method='trust-constr', args=(eps),
                           constraints=[linear_constraint], options={'verbose': 0})  # Set verbose=0 for clean output

            # Combine boundaries and optimized intermediate steps
            lambda_res = torch.tensor(np.concatenate((np.array([lambda_T]), res.x, np.array([lambda_eps]))))
            t_res = self.inverse_lambda(lambda_res).detach().clone()

        # Return the optimized timesteps (t) and lambdas
        return t_res, lambda_res