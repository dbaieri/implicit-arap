import torch
import torch.nn.functional as F

from torch import Tensor
from jaxtyping import Float

from iarap.utils.diffop import cross_skew_matrix


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


def euler_to_rotation(euler: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3 3"]:
	coords = [f/2.0 for f in torch.split(euler, 1, dim=-1)]
	cx, cy, cz = [torch.cos(f) for f in coords]
	sx, sy, sz = [torch.sin(f) for f in coords]

	quaternion = torch.cat([
		cx*cy*cz - sx*sy*sz, cx*sy*sz + cy*cz*sx,
		cx*cz*sy - sx*cy*sz, cx*cy*sz + sx*cz*sy
	], dim=-1)

	norm_quat = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)
	w, x, y, z = torch.split(norm_quat, 1, dim=-1)

	B = quaternion.shape[:-1]

	w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
	wx, wy, wz = w * x, w * y, w * z
	xy, xz, yz = x * y, x * z, y * z

	rot_mat = torch.stack([
		w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
		2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
		2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
	], dim=-1).view(*B, 3, 3)
	return rot_mat

def align_vectors(a, b):
	a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
	I = torch.diag_embed(torch.ones_like(a))
	v = torch.linalg.cross(a, b, dim=-1)
	c = (b * a).sum(dim=-1)
	cross_matrix = cross_skew_matrix(v)
	scale = ((1 - c) / v.norm(dim=-1).pow(2)).view(*cross_matrix.shape[:-2], 1, 1)
	return I + cross_matrix + (cross_matrix @ cross_matrix) * scale