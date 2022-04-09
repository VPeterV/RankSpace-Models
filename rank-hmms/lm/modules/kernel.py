import math
import torch
# from genbmm import logbmm
from ..helpers.utils import checkpoint

def get_projection(num_rows, num_columns, scaling=0):
	"""
	num rows: num features
	num columns: hidden dims
	ref code: https://github.com/justinchiu/low-rank-models/blob/9fcffc93ce2816415423bd6d24a9ff75e4e5f2a9/lhmm/models/linear_utils.py#L160
	"""
	num_blocks = num_rows // num_columns
	block_list = []
	for _ in range(num_blocks):
		block = torch.randn(num_columns, num_columns)
		q, _ = torch.qr(block)
		q = q.T
		block_list.append(q)
	remaining_rows = num_rows - num_blocks * num_columns
	if remaining_rows > 0:
		block = torch.randn(num_columns, num_columns)
		q, _ = torch.qr(block)
		q = q.T
		block_list.append(q[0:remaining_rows])
	projection = torch.cat(block_list, 0)
	
	if scaling == 0:
		multiplier = torch.norm(
			torch.randn(num_rows, num_columns), dim = -1).view(-1,1)
	elif scaling == 1:
		multiplier = math.sqrt(float(num_columns)) * torch.ones((num_rows)).view(-1,1)
	else:
		raise ValueError("scaling must be one of {0,1}. Was %s" % scaling)
		
	return multiplier * projection

def nonnegative_softmax_kernel_feature_creator(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    is_query: bool,
    eps: float=0.0001,
    log = False,
    no_shift = False,
):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """

    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    #ratio = 1.0

    bsz = data.size(0)
   
    projection = projection_matrix.unsqueeze(0).expand(bsz, -1, -1)

    # Compute wx
    # data:       bsz, len, D
    # projection: bsz, D, #features
    data_dash = torch.bmm(
        data,
        projection
    ) # bsz, len, #features

    # Compute ||x||^2/2
    # diag_data = torch.square(data)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, -1) # (bsz, len) ||x||^2
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(-1) # bsz, len, 1

    if log:
        if no_shift:
            return data_dash - diag_data
            #return math.log(ratio) + data_dash - diag_data + math.log(eps)

        if is_query:
            # test
            stuff = math.log(ratio) + data_dash - diag_data
            return stuff - stuff.max(dim=-1, keepdim=True)[0].detach()
            # /test
            # looks like the above is fine and equivalent to no_shift
            return (math.log(ratio) + data_dash - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True)[0].detach()
                #- torch.max(data_dash, dim=-1, keepdim=True)[0]
                #+ math.log(eps)
            )
        else:
            # test
            stuff = math.log(ratio) + data_dash - diag_data
            return stuff - stuff.max().detach()
            # /test
            # looks like the above is fine and equivalent to no_shift
            return (math.log(ratio) + data_dash - diag_data
                - torch.max(data_dash).detach()
                #- torch.max(data_dash)
                #+ math.log(eps)
            )

    # Compute exp(wx - ||x||^2/2)  
    # (Lemma 1, SM(x, y) = E_{w~N(0,I)} exp(wx - ||x||^2/2) exp(wy - ||y||^2/2))
    if is_query:
        # for each query, we can independently scale to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True)[0]) + eps)
    else:
        # for keys, we need to use the same normalizer to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash

@checkpoint
def mylogbmm(x, y):
    expand = x[:,:,None,:] + y[:,None,:,:]
    return expand.logsumexp(-1)

def project_logits(
    query, key, projection_matrix,
    eps=0.0001, rff_method="log", no_shift=False,
    fast=True,
):
    kernel = nonnegative_softmax_kernel_feature_creator

    if rff_method == "log":
        # log space
        log_query_features = kernel(
            query, projection_matrix, is_query=True, eps=eps, log=True,
            no_shift = no_shift,
        )
        log_key_features = kernel(
            key, projection_matrix, is_query=False, eps=eps, log=True,
            no_shift = no_shift,
        )
        #import pdb; pdb.set_trace()
        # slow and memory...would like log-bmm
        # bxz x src x tgt x dim

        if fast:
            return mylogbmm(log_query_features, log_key_features)
        # else:
            # return logbmm(log_query_features, log_key_features.transpose(-1, -2).contiguous())

    else:
        raise ValueError(f"Invalid rff_method: {rff_method}")