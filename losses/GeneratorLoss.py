import torch
from torch import nn as nn
import lpips
from histogram_matching import cal_hist, cal_trans, histogram_matching

def FinalLoss(nn.Module):
	def __init__(self, losses: List[nn.Module], coefs: List[nn.Module]):
		super().__init__()
		self.losses = nn.ModuleList(losses)
        self.coefs = torch.tensor(coefs)

    def forward(self, **kwargs):
    	result = 0.
    	for loss_func, coef in zip(self.losses, self.coefs):
    		result += loss_func(**kwargs) * coef
    	return result

# kwargs --> 'fake_disc_out', 'source', 'target', 'side', 'final'


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_disc_out: Tensor, **kwargs):
        labels = torch.ones_like(fake_disc_out).to(fake_disc_out)
        return self.bce_loss(fake_disc_out, labels)

class IdentityPreservationLoss(nn.Module):
	def __init__(self, ArcFace):
		self.model = ArcFace
		self.loss = nn.CosineEmbeddingLoss()

	def forward(self, final: Tensor, source: Tensor, **kwargs):
		with torch.no_grad():
			source_emb = self.model(source)
			final_emb = self.model(final)
		labels = torch.full((final_emb.size(0),), 1)
		return self.loss(source_emb, final_emb, labels)

class LandmarkAlignmentLoss(nn.Module):
	def __init__(self, LandmarkEstimator):
		super.__init__()
		self.model = LandmarkEstimator
		self.loss = nn.MSELoss(reduction='sum')

	def forward(self, side: Tensor, final: Tensor, target: Tensor, **kwargs):
		with torch.no_grad():
			side_emb = self.model(side)
			final_emb = self.model(final)
			target_emb = self.model(target)
		return self.loss(side_emb, target_emb) + self.loss(final_emb, target_emb)

class ReconstructionLoss(nn.Module):
	def __init__(self, alpha = 0.8):
		super().__init__()
		self.alpha = alpha
		self.lpips = lpips.LPIPS(net='vgg')
		self.loss = nn.MSELoss(reduction='sum')

	def forward(self, source: Tensor, target: Tensor, side: Tensor, final: Tensor, **kwargs):
		return self.loss(final, target) + self.loss(side, target) + self.alpha(
			self.lpips(final, target) + self.lpips(side, target))

class StyleTransferLoss(nn.Module):
	def __init__(self)
		super().__init__()
 
	def mask_preprocess(self, mask):
        index_tmp = mask.nonzero()
        x_index = index_tmp[:, 2]
        y_index = index_tmp[:, 3]
        return [x_index, y_index, x_index, y_index]

	def forward(self, target: Tensor, final: Tensor, t_mask: Tensor, **kwargs):
		index = self.mask_index(mask)
		HM = histogram_matching(final, target, index) #need to define index
    	return np.linalg.norm(yf - HM)


