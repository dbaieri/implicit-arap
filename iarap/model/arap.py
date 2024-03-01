
import torch
import kaolin as kal
import torch.nn.functional as F

from tqdm import tqdm
from kaolin.rep import SurfaceMesh

from iarap.utils import cholesky_invert, least_sq_with_known_values, qr_invert


class ARAPMesh(SurfaceMesh):
    """
    Subclass of PyTorch3D Meshes.
    Upon deformation, provided with 'static' vertices, and 'handle' vertices.
    Allows for movement of handle vertices from original mesh position, and identifies new positions of all other
    non-static verts using the As Rigid As Possible algorithm (As-Rigid-As-Possible Surface Modeling, O. Sorkine & M. Alexa)"""

    def __init__(self, verts=None, faces=None, device='cpu', internal_scaling: float=1):
        """
        lists of verts, faces and textures for methods. For details, see Meshes documentation

        :param optimise: flag if mesh will be used for optimisation
        """
        super(ARAPMesh, self).__init__(verts.to(device) * internal_scaling, faces.to(device))

        self.internal_scaling = internal_scaling
        self.adjacency = kal.ops.mesh.adjacency_matrix(self.vertices.shape[0], self.faces).coalesce()
        self.one_ring = self.get_one_ring_neighbours()
        self.neighbor_counts = self.adjacency.sum(dim=0).long().to_dense()
        self.max_neighbors = self.neighbor_counts.max().item()
        self.first_edge_idx = F.pad(self.neighbor_counts.cumsum(dim=0)[:-1], (1, 0)).long()
        self.C = self.vertices.mean(dim=0)  # centre of mass
        self.precomputed_params = {}  # dictionary to store precomputed parameters

        # Precompute cotangent weights in nfmt. nfmt is defined at the top of this script
        self.device = device
        # w_full = get_cot_weights_full(self, device=self.device)
        # self.w_nfmts = produce_cot_weights_nfmt(self, self.one_ring, device=self.device)
        
    def get_one_ring_neighbours(self):
        return self.adjacency.indices()

    def get_flat_edge_index(self):
        orn = self.one_ring
        V = self.vertices.shape[0]
        ii = torch.arange(0, V, device=self.device).unsqueeze(-1).expand(-1, self.max_neighbors)
        jj = torch.arange(0, V, device=self.device).unsqueeze(-1).repeat(1, self.max_neighbors)
        nn = torch.arange(0, self.max_neighbors, device=self.device).unsqueeze(0).expand(V, -1)
        for k in range(self.max_neighbors):
            all_kth_edges = orn[1, torch.clamp(self.first_edge_idx + k, max=orn.shape[1]-1)]
            jj[self.neighbor_counts >= (k+1), k] = all_kth_edges[self.neighbor_counts >= (k+1)]
        return ii.reshape(-1), jj.reshape(-1), nn.reshape(-1)
    
    def get_nfmt_edge_matrix(self, verts, ii, jj, nn):
        E = torch.zeros(verts.shape[0], self.max_neighbors, 3).to(self.device)
        E[ii, nn] = verts[ii] - verts[jj]
        return E
        
    def get_cot_weights_pyg(self, eps=1e-12) -> torch.Tensor:
        """Retrieves cotangent weights 0.5 * cot a_ij + cot b_ij for each mesh. Returns as a padded tensor.

        :return cot_weights: Tensor of cotangent weights, shape (N_mesh, max(n_verts_per_mesh), max(n_verts_per_mesh))
        :type cot_weights: Tensor"""
        verts, faces = self.vertices, self.faces
        V, F = verts.shape[0], faces.shape[0]

        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
  
        idx = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0).T

        def get_cots(A, B, C):
            left_vec = A - B
            right_vec = C - B
            dot = torch.einsum('ij, ij -> i', left_vec, right_vec)
            cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
            cot = dot / cross  # cot = cos / sin
            return cot / 2.0  # by definition

        # For each triangle face, get all three cotangents:
        cot_A = get_cots(v0, v2, v1)
        cot_B = get_cots(v1, v0, v2)
        cot_C = get_cots(v0, v1, v2)
        cot = torch.cat([cot_A, cot_B, cot_C])

        w = torch.sparse_coo_tensor(idx, cot.view(-1), (V, V))  # This doesn't look right
        w = (w + w.t()).to_dense()
        L = torch.diag(torch.sum(w, dim=0)) - w

        '''
        def get_areas(A, B, C):
            left_vec = A - B
            right_vec = C - B
            cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
            area = cross / 6.0  # one-third of a triangle's area is cross / 6.0
            return area / 2.0  # since each corresponding area is counted twice

        area_A = get_areas(v0, v2, v1)
        area_B = get_areas(v1, v0, v2)
        area_C = get_areas(v0, v1, v2)
        area = torch.cat([area_A, area_B, area_C])

        # For each vertex, compute the sum of areas for triangles containing it.
        areas = torch.sparse_coo_tensor(idx, area.view(-1), (V, V))
        areas = areas + areas.t()
        inv_areas = areas.sum(dim=0).to_dense().pow(-0.5)
        inv_areas[inv_areas.isinf()] = 0.0
        area_idx = L.nonzero().T  
        inv_areas = torch.sparse_coo_tensor(area_idx, inv_areas[area_idx[0, :]] * inv_areas[area_idx[1, :]], (V, V))
        '''

        return L, w  #  * inv_areas.to_dense() , w
    
    def get_cot_weights_p3d(self, eps=1e-12) -> torch.Tensor:
        """Retrieves cotangent weights 0.5 * cot a_ij + cot b_ij for each mesh. Returns as a padded tensor.

        :return cot_weights: Tensor of cotangent weights, shape (N_mesh, max(n_verts_per_mesh), max(n_verts_per_mesh))
        :type cot_weights: Tensor"""
        verts, faces = self.vertices, self.faces
        V, F = verts.shape[0], faces.shape[0]

        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        
        # Side lengths of each triangle, of shape (sum(F_n),)
        # A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
        s = 0.5 * (A + B + C)
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

        # Compute cotangents of angles, of shape (sum(F_n), 3)
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0

        ii = faces[:, [1, 2, 0]]
        jj = faces[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
        w = torch.sparse_coo_tensor(idx, cot.view(-1), (V, V))

        # Make it symmetric; this means we are also setting
        w = (w + w.t()).to_dense()
        L = torch.diag(torch.sum(w, dim=0)) - w

        # For each vertex, compute the sum of areas for triangles containing it.
        # idx = faces.view(-1)
        # inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
        # val = torch.stack([area] * 3, dim=1).view(-1)
        # inv_areas.scatter_add_(0, idx, val)
        # idx = inv_areas > 0
        # inv_areas[idx] = 1.0 / inv_areas[idx]
        # inv_areas = inv_areas.view(-1, 1)

        return L, w

    def precompute_nfmt_cot_weights(self, w, ii=None, jj=None, nn=None):
        V = self.vertices.shape[0]

        Wn = torch.zeros((V, self.max_neighbors)).to(self.device)
        if None in {ii, jj, nn}:
            ii, jj, nn = self.get_flat_edge_index()

        Wn[ii, nn] = w[ii, jj]
        self.precomputed_params['w_nfmt'] = Wn

    def precompute_laplacian(self):
        """Precompute edge weights and Laplacian-Beltrami operator"""
        # L, w = self.get_cot_weights_pyg()
        L, w = self.get_cot_weights_p3d()

        self.precomputed_params["L"] = L
        self.precomputed_params["w"] = w
        
    def precompute_reduced_laplacian(self, static_verts, handle_verts):
        """Precompute the Laplacian-Beltrami operator for the reduced set of vertices, negating static and handle verts"""
        L = self.precomputed_params["L"]
        n = self.vertices.shape[0]
        unknown_mask = torch.ones(n, dtype=torch.bool, device=self.device)  # all unknown verts
        unknown_mask[static_verts] = 0
        unknown_mask[handle_verts] = 0
        L_reduced = L[unknown_mask, :][:, unknown_mask]  # sample sub laplacian matrix for unknowns only
        L_reduced_inv = qr_invert(L_reduced)  
        self.precomputed_params["L_reduced_inv"] = L_reduced_inv

    def compute_energy(self, 
                       verts_deformed: torch.Tensor,
                       ii=None, jj=None, nn=None):
        """Compute the energy of a deformation for a deformation, according to

        sum_i w_i * sum_j w_ij || (p'_i - p'_j) - R_i(p_i - p_j) ||^2

        Where i is the vertex index,
        j is the indices of all one-ring-neighbours
        p gives the undeformed vertex locations
        p' gives the deformed vertex rotations
        and R gives the rotation matrix between p and p' that captures as much of the deformation as possible
        (maximising the amount of deformation that is rigid)

        w_i gives the per-cell weight, selected as 1
        w_ij gives the per-edge weight, selected as 0.5 * (cot (alpha_ij) + cot(beta_ij)), where alpha and beta
        give the angles opposite of the mesh edge

        :param verts_deformed:

        :return energy: Tensor of strain energy of deformation

        """

        if None in {ii, jj, nn}:
            ii, jj, nn = self.get_flat_edge_index()

        w = self.precomputed_params['w_nfmt']  # cotangent weight matrix, in nfmt index format

        p = self.vertices  # initial mesh
        p_prime = verts_deformed  # displaced verts

        P = self.get_nfmt_edge_matrix(p, ii, jj, nn)
        P_prime = self.get_nfmt_edge_matrix(p_prime, ii, jj, nn)

        ### Calculate covariance matrix in bulk
        D = torch.diag_embed(w, dim1=1, dim2=2)
        S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))

        ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
        unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=1))[0])  # any verts which are undeformed
        S[unchanged_verts] = 0

        U, sig, W = torch.svd(S)
        R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations

        # Need to flip the column of U corresponding to smallest singular value
        # for any det(Ri) <= 0
        entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
        if len(entries_to_flip) > 0:
            Umod = U.clone()
            cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
            Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
            R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))

        # Compute energy
        rot_rigid = torch.bmm(R, P.permute(0, 2, 1)).permute(0, 2, 1)
        stretch_vec = P_prime - rot_rigid  # stretch vector
        stretch_norm = (torch.norm(stretch_vec, dim=2) ** 2)  # norm over (x,y,z) space
        energy = (w * stretch_norm).sum()

        return energy

    def solve(self, 
              static_verts, 
              handle_verts, 
              handle_verts_pos, 
              n_its=1,
              track_energy=False, 
              report=False):
        """
        Solve iterations of the As-Rigid-As-Possible method.

        :param static_verts: list of all vertices which do not move
        :param handle_verts: list of all vertices which are moved as input. Size H
        :param handle_verts_pos: (H x 3) array of target positions of all handle_verts
        :param mesh_idx: index of self for selected mesh.
        :param track_energy: Flag to print energy after every it
        :param report: Flag to use tqdm bar to track iteration progress

        p = initial mesh deformation
        p0 = working guess
        """

        handle_verts_pos = handle_verts_pos.to(self.device)
        V = self.vertices.shape[0]
        p = self.vertices  # initial mesh

        if "w" not in self.precomputed_params:
            self.precompute_laplacian()

        L = self.precomputed_params["L"].to(self.device)

        known_mask = torch.zeros(V, dtype=torch.bool, device=self.device)
        known_mask[handle_verts] = 1
        known_mask[static_verts] = 1
        known_val = torch.zeros(V, 3, device=self.device)
        known_val[static_verts, :] = p[static_verts]
        known_val[handle_verts, :] = handle_verts_pos

        # Initial guess using Naive Laplacian editing: least square minimisation of |Lp0 - Lp|, subject to known
        # constraints on the values of p, from static and handles
        p_prime = least_sq_with_known_values(L, torch.mm(L, p), known_mask=known_mask, known_val=known_val)

        if n_its == 0:  # if only want initial guess, end here
            return p_prime

        ## modify L, L_inv and b_fixed to incorporate boundary conditions
        unknown_verts = ~known_mask  # indices of all unknown verts

        b_fixed = torch.zeros((V, 3), device=self.device)  # factor to be subtracted from b, due to constraints
        # for k, pos in known.items():
        b_fixed += torch.einsum("nm,mf->nf", L[:, known_mask], known_val[known_mask, :])  # [unknown]

        #  Precompute L_reduced_inv if not already done
        if "L_reduced_inv" not in self.precomputed_params:
            self.precompute_reduced_laplacian(static_verts, handle_verts)
        L_reduced_inv = self.precomputed_params["L_reduced_inv"].to(self.device)

        ii, jj, nn = self.get_flat_edge_index()  # flattened tensors for indices
        
        if "w_nfmt" not in self.precomputed_params:
            self.precompute_nfmt_cot_weights(self.precomputed_params['w'], ii, jj, nn)
        w = self.precomputed_params['w_nfmt']  # cotangent weight matrix, in nfmt index format

        P = self.get_nfmt_edge_matrix(self.vertices, ii, jj, nn)

        # Iterate through method
        if report: progress = tqdm(total=n_its)

        for it in range(n_its):

            P_prime = self.get_nfmt_edge_matrix(p_prime, ii, jj, nn)

            ### Calculate covariance matrix in bulk
            D = torch.diag_embed(w, dim1=1, dim2=2)
            S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))

            ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
            unchanged_verts = torch.unique(torch.where((P == P_prime).all(dim=1))[0])  # any verts which are undeformed
            S[unchanged_verts] = 0

            U, sig, W = torch.svd(S)
            R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations

            # Need to flip the column of U corresponding to smallest singular value
            # for any det(Ri) <= 0
            entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
            if len(entries_to_flip) > 0:
                Umod = U.clone()
                cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
                Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
                R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))

            ### RHS of minimum energy equation
            Rsum_shape = (V, self.max_neighbors, 3, 3)
            Rsum = torch.zeros(Rsum_shape).to(self.device)  # Ri + Rj, as in eq (8)
            Rsum[ii, nn] = R[ii] + R[jj]

            ### Rsum has shape (V, max_neighbours, 3, 3). P has shape (V, max_neighbours, 3)
            ### To batch multiply, collapse first 2 dims into a single batch dim
            Rsum_batch, P_batch = Rsum.view(-1, 3, 3), P.view(-1, 3).unsqueeze(-1)

            # RHS of minimum energy equation
            b = 0.5 * (w[..., None] * torch.bmm(Rsum_batch, P_batch).squeeze(-1).reshape(V, self.max_neighbors, 3)).sum(dim=1)

            b -= b_fixed  # subtract component of LHS not included - constraints

            p_prime_unknown = torch.mm(L_reduced_inv, b[unknown_verts])  # predicted p's for only unknown values

            p_prime = torch.zeros_like(p_prime)  # generate next iteration of fit, from p0_unknown and constraints
            p_prime[known_mask, :] = known_val[known_mask, :]

            # Assign initially unknown values to x_out
            p_prime[unknown_verts] = p_prime_unknown

            # track energy
            if track_energy:
                energy = self.compute_energy(p_prime, ii, jj, nn)
                print(f"It = {it}, Energy = {energy:.5f}")
            # update tqdm
            if report:
                progress.update()

        return p_prime / self.internal_scaling # return new vertices

    def debug_patches(self, pts_per_patch, num_patches):
        out = []
        self.precompute_laplacian()
        L = self.precomputed_params['L']
        for i in range(num_patches):
            L_i = L[i*pts_per_patch:(i+1)*pts_per_patch, i*pts_per_patch:(i+1)*pts_per_patch]
            try:
                cholesky_invert(L_i)
            except:
                out.append(i)
        return out