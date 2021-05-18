# time: 2021/5/14 22:28
# File: crf.py
# Author: zhangshubo
# Mail: supozhang@126.com
import torch
from torch.nn.parameter import Parameter
from torch.nn import init


class CRFLoss(torch.nn.Module):
    def __init__(self):
        super(CRFLoss, self).__init__()

    def forward(self, crf, y_true, y_pred, mask):
        return crf.crf_loss(y_true, y_pred, mask=mask)


class CRF(torch.nn.Module):

    def __init__(self, units, learn_mode, test_mode, device="cpu"):
        """
        :param units: 输出维度
        :param learn_mode: `join` 或者 `marginal`
        :param test_mode:  `viterbi` 或者 `marginal`
        """
        super(CRF, self).__init__()
        self.units = units
        self.learn_mode = learn_mode
        self.test_mode = test_mode
        self.transition = Parameter(torch.Tensor(units, units),  requires_grad=True)
        init.xavier_normal(self.transition)
        self.device = device
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def reset_parameters(self):
        init.xavier_normal(self.transition)

    def forward(self, x, mask=None):
        if self.learn_mode == 'viterbi':
            return x

    def step(self, input_energy_t, last_state, mask=None):
        """
        B - Batch size
        F - Final dim or output dim
        last_state - B*F
        input_energy_t - B*F
        """
        if mask is not None:
            input_energy_t = input_energy_t * mask[:, 1].unsqueeze(-1)
        input_energy_t = input_energy_t .unsqueeze(1)  # B * 1 * F

        last_state = last_state.unsqueeze(2)  # B * F * 1
        transition = self.transition.unsqueeze(0)  # 1 * F * F
        chain_energy = input_energy_t + transition

        if mask is not None:
            mask = mask[:, 0] * mask[:, 1]
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            chain_energy = chain_energy * mask
        score = last_state + chain_energy

        new_state = self._log_sum_exp(score, 1)  # (batch_size, num_tags)
        return new_state

    def get_negative_log_likelihood(self, y_true, x, mask=None):
        """
        计算负极大似然对数函数，之所以是负的，是为了将极大似然转化为求最小值，也就是通常意义loss
        likelihood = 1/Z * exp(E) neg_log_like = - log(1/Z*exp(E)) = Z-E
        """
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)
            y_true = torch.zeros(
                y_true.size(0), y_true.size(1), self.units).to(self.device).scatter_(2, y_true, 1)
        sequence_len = x.size(1)
        energy = self.get_energy(y_true, x, mask=mask)
        hidden_state = x[:, 0, :]
        for i in range(1, sequence_len):
            hidden_state = self.step(x[:, i, :], hidden_state, mask=mask[:, i-1:i+1])

        nll = self._log_sum_exp(hidden_state, 1) - energy
        if mask is not None:
            nll = nll/torch.sum(mask, 1)
        else:
            nll = nll/sequence_len
        return torch.sum(nll)/y_true.size(0)

    def crf_loss(self, y_true, x, mask=None):

        if self.learn_mode == "join":
            return self.get_negative_log_likelihood(y_true, x, mask=mask)
        else:
            return self.cross_entropy(x.view(-1, self.units), y_true.view(-1))

    def get_energy(self, y_true, input_energy, mask):
        dim2 = y_true.size(2)
        dim1 = y_true.size(1)
        input_energy = torch.sum(input_energy * y_true, 2)  # B * T
        # y_true is B * T * F, transition F * F
        # 选择从T-1开始的转移概率
        _ = torch.mm(y_true[:, :-1, :].reshape(-1, dim2), self.transition).reshape(-1, dim1-1, dim2)
        # 选择到T的转移概率，然后降维
        chain_energy = torch.sum(_ * y_true[:, 1:, :], 2)

        if mask is not None:
            chain_mask = mask[:, :-1] * mask[:, 1:]
            input_energy = input_energy * mask
            chain_energy = chain_energy * chain_mask
        total_energy = torch.sum(input_energy, -1) + torch.sum(chain_energy, -1)
        return total_energy

    @staticmethod
    def _log_sum_exp(tensor, dim):
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
        # Add offset back
        return offset + safe_log_sum_exp

    def viterbi_decoding(self, x):
        sequence_len = x.size(1)
        hidden_state = x[:, 0, :]
        paths = []
        for i in range(1, sequence_len):
            input_energy_t = x[:, i, :].unsqueeze(1)  # B * 1 * F
            hidden_state = hidden_state.unsqueeze(2)  # B * F * 1
            transition = self.transition.unsqueeze(0)  # 1 * F * F
            # 这里可以不用求log-sum-exp了，因为单调性是相同的
            score = hidden_state + input_energy_t + transition
            best_score, best_path = score.max(1)
            hidden_state = best_score
            paths.append(best_path)
        # 找到最后一列的最优序号
        _, last_best = hidden_state.max(1)
        final_path = [last_best]
        last_best = last_best.unsqueeze(-1)
        # 回溯之前的最优序号
        for p in paths[::-1]:
            last_best = torch.gather(p, 1, last_best)
            final_path.append(list(last_best.squeeze(-1).cpu().numpy()))
        path = torch.tensor(final_path[::-1]).transpose(1, 0)
        return path




