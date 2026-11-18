from torch import Tensor,tensor
from typing import Optional
from torch import nn
import torch

class MIScore(nn.Module):
  def __init__(self,bins,min,max):
    super(MIScore, self).__init__()
    self.bins = bins
    self.min = min
    self.max = max
    self.divide = torch.tensor((self.max - self.min) / (self.bins))

  def calculate_contingency_matrix(self,preds: Tensor, target: Tensor, eps: Optional[float] = None, sparse: bool = False
  ) -> Tensor:

    preds_classes, preds_idx = torch.unique(preds, return_inverse=True)
    target_classes, target_idx = torch.unique(target, return_inverse=True)

    num_classes_preds = preds_classes.size(0)
    num_classes_target = target_classes.size(0)

    contingency = torch.sparse_coo_tensor(
        torch.stack(
            (
                target_idx,
                preds_idx,
            )
        ),
        torch.ones(target_idx.shape[0], dtype=preds_idx.dtype, device=preds_idx.device),
        (
            num_classes_target,
            num_classes_preds,
        ),
    )

    if not sparse:
        contingency = contingency.to_dense()
        if eps:
            contingency = contingency + eps

    return contingency
  def _mutual_info_scorec_compute(self,contingency: Tensor) -> Tensor:
      """Compute the mutual information score based on the contingency matrix.

      Args:
          contingency: contingency matrix

      Returns:
          mutual_info: mutual information score

      """
      n = contingency.sum()
      u = contingency.sum(dim=1)
      v = contingency.sum(dim=0)

      # Check if preds or target labels only have one cluster
      if u.size() == 1 or v.size() == 1:
          return tensor(0.0)

      # Find indices of nonzero values in U and V
      nzu, nzv = torch.nonzero(contingency, as_tuple=True)
      contingency = contingency[nzu, nzv]

      # Calculate MI using entries corresponding to nonzero contingency matrix entries
      log_outer = torch.log(u[nzu]) + torch.log(v[nzv])
      mutual_info = contingency / n * (torch.log(n) + torch.log(contingency) - log_outer)
      return mutual_info.sum()

  def forward(self,targs: Tensor,preds: Tensor) -> Tensor:
    targs_norm = targs.ravel() // self.divide
    preds_norm = preds.ravel() // self.divide
    contingency = self.calculate_contingency_matrix(preds = preds_norm, target=targs_norm)
    score = self._mutual_info_scorec_compute(contingency)
    return score