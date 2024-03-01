import torch


def least_sq_with_known_values(A, b, known_mask=None, known_val=None):
	"""Solves the least squares problem minx ||Ax - b||2, where some values of x are known.
	Works by moving all known variables from A to b.

	:param A: full rank matrix of size (m x n)
	:param b: matrix of size (m x k)
	:param known: dict of known_variable : value

	:type A: torch.Tensor
	:type B: torch.Tensor
	:type known: dict

	:returns x: matrix of size (n x k)
	"""

	M, N = A.shape
	M2, K = b.shape
	assert M == M2, "A's first dimension must match b's"

	if known_mask is None: 
		known_mask = torch.zeros([M2,], dtype=torch.bool, device=A.device)
		known_val = torch.empty([0, K], device=A.device)

	# Move to b
	col = A[:, known_mask]
	b[known_mask] -= torch.einsum("ni,nj->ij", col, known_val)

	# Remove from A
	unknown = ~known_mask
	A = A[:, unknown]  # only continue with cols for unknowns

	x = torch.linalg.lstsq(b, A).solution

	# all unknown values have now been found. Now create the output tensor, of known and unknown values in correct positions
	x_out = torch.zeros((N, K), device=A.device)
	x_out[known_mask, :] = known_val[known_mask, :]
	x_out[unknown, :] = x.T
	
	return x_out

def cholesky_invert(A):
	L = torch.linalg.cholesky(A)
	L_inv = torch.inverse(L)
	A_inv = torch.mm(L_inv.T, L_inv)
	return A_inv

def qr_invert(A):
	Q, R = torch.linalg.qr(A)
	return torch.inverse(R) @ Q.mT