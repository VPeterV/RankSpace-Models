
import torch
from torch import nn

from lm.helpers.utils import Pack

import time as timep
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from ..hmm.ranklogbmm import forward

from lm.modules.layers import LogDropout
from lm.modules.neural_modules import WordMLP, WordMLP_H

from lm.helpers.utils import Pack, checkpoint, logspace
from lm.modules.kernel import get_projection

class ProjRankSpace(nn.Module):
	def __init__(self, V, args):
		super(ProjRankSpace, self).__init__()
		
		config = args.model     # model config
		self.config = config
		self.V = V
		self.device = args.device

		self.C = config.num_classes

		self.timing = args.timing
		self.is_chp = args.timing
			
		self.transmlp = config.transmlp
		
		self.projection = nn.Parameter(torch.randn(config.rank, config.hidden_dim))

		# p(z0)
		# |Z| x h
		self.start_emb = nn.Parameter(torch.randn(self.C, config.hidden_dim))

		self.start_mlp = WordMLP(config.hidden_dim,
										config.hidden_dim, 1, dropout=config.dropout)

		# p(zt | zt-1)
		# A (b,t,zt-1,zt) => U(b,t,zt-1,r) X(b,t,zt,r)
		self.state_emb = nn.Parameter(torch.randn(self.C, config.hidden_dim))
		
		if self.transmlp:
			self.l_transition_mlp = WordMLP_H(config.hidden_dim,
											config.hidden_dim)
			self.r_transition_mlp = WordMLP_H(config.hidden_dim,
											  config.hidden_dim)

		self.next_state_emb = nn.Parameter(torch.randn(self.C, config.hidden_dim))

		# p(xt | zt)
		self.vocab_emb = nn.Parameter(torch.randn(len(V), config.hidden_dim))

		self.vocab_emission_mlp = WordMLP(config.hidden_dim,
										config.hidden_dim, config.rank, dropout=config.dropout) # |V| x h => |V| x d


		self.dropout = nn.Dropout(config.dropout)

		self.states_dropout = config.states_dropout
		self.rank_dropout = config.rank_dropout
		
		self.logdropout = LogDropout(config.dropout, self.device)
		
		self.rank = config.rank

	def start(self, start_dropout, states=None, emb=None):
		start_emb = self.start_emb if start_dropout is None else self.start_emb(states)[~start_dropout]
		
		if emb is not None:
			start = self.start_mlp(emb)
		else:
			start = self.start_mlp(self.dropout(start_emb))
			
			
		start = start.squeeze(-1).log_softmax(-1)


		return start

	def transition_logits(self, states_dropout, rank_dropout, states=None, emb=None):
		# get transition matrix
		
		state_emb = self.state_emb if states_dropout is None else self.state_emb(states)[~states_dropout]
		next_state_emb = self.next_state_emb if states_dropout is None else self.next_state_emb(states)[~states_dropout]
		projection = self.projection.t() if rank_dropout is None else self.projection.t()[:, ~rank_dropout]
		
		if self.transmlp:
			state_emb = self.l_transition_mlp(state_emb)
			next_state_emb = self.r_transition_mlp(next_state_emb)
		

		loglt = state_emb @ projection
		logrt = next_state_emb @ projection
		
		loglt = self.logdropout(loglt)
		logrt = self.logdropout(logrt)

		return loglt, logrt

	def normalize_transition(self, loglt, logrt):
		
		return loglt.log_softmax(-1), logrt.log_softmax(-2)	# raw-normalized, column-normalized

	def emission_logits(self, rank_dropout, states=None):
		# get emission tensor
		vocab_emb = self.vocab_emb
		vocab_emission = self.vocab_emission_mlp(vocab_emb)
		vocab_emission = vocab_emission
		if rank_dropout is not None:
			vocab_emission = vocab_emission[:, ~rank_dropout]

		return vocab_emission

	def normalize_emission(self, v_emission):

		return v_emission.log_softmax(-2)

	def compute_parameters(self, start_dropout, states_dropout, rank_dropout, text, states=None,
						   lpz=None, last_states=None,
						   ):
		start_dropout = start_dropout.to(torch.float32) if start_dropout is not None else None
		states_dropout = states_dropout.to(torch.float32) if states_dropout is not None else None
		rank_dropout = rank_dropout.to(torch.float32) if rank_dropout is not None else None
		
		batch = text.size(0)

		l_transition, r_transition = self.transition_logits(states_dropout, rank_dropout, states)
		
		loglt, logrt = self.normalize_transition(l_transition, r_transition)


		start_states = None
		start = self.start(start_dropout, start_states)		# the start symbol

		ve = self.emission_logits(rank_dropout, states)
		vocab_emission = self.normalize_emission(ve)

		return start, loglt, logrt, vocab_emission

	def log_potentials(
			self, text, states=None,
			lpz=None, last_states=None,
	):
		m =  (torch.empty(self.C, device=self.device).bernoulli_(self.states_dropout).bool()) if self.states_dropout > 0 and self.training else None
		states_dropout, start_dropout = m, m 
		rm =  (torch.empty(self.rank, device=self.device).bernoulli_(self.rank_dropout).bool()) if self.rank_dropout > 0 and self.training else None
		rank_dropout = rm

		start, loglt, logrt,vocab_emission \
			= self.compute_parameters( start_dropout, states_dropout, rank_dropout,
			text, states, lpz, last_states,
		)

		logobs = vocab_emission[text,:]
		bsz = logobs.size(0)

		init = start.unsqueeze(0).expand(bsz,-1)
		if lpz is not None:
			init = lpz.squeeze(-2)

		logrank_pi, rankt = self.clamp(text, init, loglt, logrt)

		return logrank_pi, rankt, logobs, logrt

	def clamp(
			self, text, init, loglt, logrt, 
	):
		""" 
		lt_logits: lt_logits
		rt_logits: rt_logits
		loglt: lt_logsoftmax
		logrt: rt_logsoftmax
		"""

		batch, time = text.shape
				
		lt = loglt.exp()
		rt = logrt.exp()
		start_lt = lt
			
		rank_pi = torch.einsum('bm,mr->br',init.exp(),start_lt)
		logrank_pi = logspace(rank_pi)
		
		rankt = torch.einsum('ij,jk->ik', rt.transpose(-2,-1), lt) # r x r
		
		return logrank_pi, rankt


	def compute_loss(
			self,
			rank_pi, rankt, logobs, logrt, text, mask, lengths,
			lpz, states=None, last_states=None,keep_counts=False,
	):
		'''
		Computing loss for eval
		Args:
			log_probs:
			lengths:
			reset:
			keep_counts:

		Returns:

		'''
		N = lengths.shape[0]

		chp = True if self.config.chp_theta > 0 else False
		
		logrank_pi, rankt, logobs, logrt = self.log_potentials(
			text,
			states,
			lpz, last_states,
		)
		
		m, logerank = forward(logrank_pi, rankt, logobs, mask, lengths)

		m = torch.einsum('br,mr->bm',m, logrt.exp())
		
		logm = logspace(m) + logerank[:,None]
		
		
		logz = logm.logsumexp(-1)
		
		evidence = logz.sum(-1)

		return Pack(loss = evidence, evidence = evidence, elbo = None), logm.log_softmax(-1).detach()

	def score(
			self, text,
			lpz=None, last_states=None,
			mask=None, logzt_mask=None, lengths=None,
	):
		'''
		for training
		'''
		N, T = text.shape
		
		states = None
		if self.timing:
			startpot = timep.time()

		logrank_pi, rankt, logobs, logrt = self.log_potentials(
			text,
			states,
			lpz, last_states,
		)
		if self.timing:
			print(f"log pot: {timep.time() - startpot}")
		
		m, logerank = forward(logrank_pi, rankt, logobs, mask, lengths)

		m = torch.einsum('br,mr->bm',m, logrt.exp())
		
		logm = logspace(m) + logerank[:,None]
		
		
		logz = logm.logsumexp(-1)
		
		evidence = logz.sum(-1)

		return Pack(loss = evidence, evidence=evidence,elbo=None), logm.log_softmax(-1).detach()