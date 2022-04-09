import torch as th
import torch.nn as nn
			
# post-LN
class SymbolLayer(nn.Module):
	def __init__(self, in_dim = 100,
					 out_dim = 100,
					 dropout = 0.,
					 ):
		super(SymbolLayer, self).__init__()
		self.lin1 = nn.Linear(in_dim, out_dim)
		self.lin2 = nn.Linear(out_dim, out_dim)
		self.relu = nn.ReLU()

	def forward(self, x):
	
		return self.relu(self.lin2(self.relu(self.lin1(x))))

		
class SeqLayer(nn.Module):
	def __init__(self, in_dim = 100,
				 hidden_dim = 100,
				 out_dim = 100,
				 dropout = 0.,
				 ):
		super(SeqLayer, self).__init__()               
		self.lin1 = nn.Linear(in_dim, hidden_dim)
		# self.mid = nn.Linear(hidden_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, out_dim)
		# self.layer_norm = nn.LayerNorm(out_dim)
		# self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.lin1(x)
		x = x.relu()
		# x = self.mid(x)
		# x = x.relu()
		x = self.lin2(x)

		return x

class ResLayer_CPCFG(nn.Module):
	def __init__(self, in_dim=100,
				 out_dim=100):
		super(ResLayer_CPCFG, self).__init__()
		self.lin1 = nn.Linear(in_dim, out_dim)
		self.lin2 = nn.Linear(out_dim, out_dim)
		self.relu = nn.ReLU()
		# self.lrelu = nn.LeakyReLU(0.1)

	def forward(self, x):
		x_1 = self.relu(self.lin1(x))
		x_2 = self.relu(self.lin2(x_1)) + x
		# x_1 = self.lrelu(self.lin1(x))
		# x_2 = self.lrelu(self.lin2(x_1)) + x
	
		return  x_2

class ResidualLayer(nn.Module):
	def __init__(
		self, in_dim = 100,
		out_dim = 100,
		dropout = 0.,
	):
		super(ResidualLayer, self).__init__()
		self.lin1 = nn.Linear(in_dim, out_dim)
		self.lin2 = nn.Linear(out_dim, out_dim)
		self.layer_norm = nn.LayerNorm(out_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.lin1(x)
		x = x.relu()
		x = self.dropout(x)
		return self.layer_norm(self.dropout(self.lin2(x).relu()) + x)		

class LogDropout(nn.Module):
	def __init__(
		self, p, device
	):
		super(LogDropout, self).__init__()
		self.p = p
		self.device = device
		
	def forward(self, x, column_dropout=False):
		if self.training and self.p > 0:
			if not column_dropout:
				annihilate_mask = th.empty_like(x).fill_(self.p).bernoulli().bool().to(self.device)
			else:
				annihilate_mask = (th.empty(x.shape[-1])
					.fill_(self.p).bernoulli().bool()[None].expand(x.shape).to(self.device)
				)
			return x.masked_fill(annihilate_mask, float("-inf"))
		else:
			return x

