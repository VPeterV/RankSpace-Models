import torch
from ..helpers.utils import logspace

def forward(logstart, rankt, logobs, mask, lengths):
    """
    ref code (a good trick to avoid intermedia varibles in logsumexp, saving memory)
    https://github.com/justinchiu/low-rank-models/blob/9fcffc93ce2816415423bd6d24a9ff75e4e5f2a9/lhmm/models/blhmm.py#L428
    """
    T = logobs.size(1)
    rank = rankt.size(-1)
    alphas_bmm = []
    evidences_bmm = []
    alpha = logstart + logobs[:, 0]
    logzt = alpha.logsumexp(-1, keepdim=True)
    alpha_scale = (alpha - logzt).exp()
    
    alphas_bmm.append(alpha_scale.unsqueeze(-2))
    evidences_bmm.append(logzt)
    
    for t in range(1, T):
        alpha = logspace(alpha_scale @ rankt) + logobs[:,t]
        logzt = alpha.logsumexp(-1, keepdim=True)
        alpha_scale = (alpha - logzt).exp()
        alphas_bmm.append(alpha_scale.unsqueeze(-2))
        evidences_bmm.append(logzt)
    
    alphas = torch.cat(alphas_bmm, -2)
    evidences = torch.cat(evidences_bmm, -1)
        
    a_len_idx = (lengths - 1)[:, None, None].expand(lengths.size(0), -1, rank)

    
    m = alphas.gather(1, a_len_idx)
    evidences = evidences.masked_fill(~mask, 0)
    logerank = evidences.sum(-1)    # evidence before the last element
    
    
    return m.squeeze(-2), logerank
    
	
