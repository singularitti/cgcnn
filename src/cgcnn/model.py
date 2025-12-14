import torch
import torch.nn as nn


__all__ = ["ConvLayer", "CrystalGraphConvNet"]


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        # Linear layer maps concatenated [center_atom(node)_fea, neighbor_atom(node)_fea, bond(edge)_fea]
        # of size (2*atom_fea_len + nbr_fea_len) → output size (2*atom_fea_len)
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        # Batch normalization over (N*M, 2F) in message update stage
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        # Batch normalization over (N, F) after aggregation
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        # N: number of center atoms (nodes).
        # M: number of neighbor atoms (nodes) per center atom (node)
        N, M = nbr_fea_idx.shape
        # convolution
        # Gather neighbor atom (node) features:
        # `atom_in_fea` shape `(N, F)`, `nbr_fea_idx` shape `(N, M)`
        # `atom_in_fea[nbr_fea_idx, :]` performs advanced indexing on axis 0:
        #   for each center atom (node) i and neighbor slot j,
        #   `nbr_fea_idx[i, j] = k` → neighbor atom (node) index k
        #   retrieves `atom_in_fea[k, :]` of shape `(F,)`
        # The resulting `atom_nbr_fea` has shape `(N, M, F)`
        # Each `atom_nbr_fea[i, j, :]` = feature vector of j-th neighbor atom (node)
        # of center atom (node) i.
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        # Combine (concatenate) the center-atom, neighbor-atom, and bond features
        # to form the full per-edge input tensor $z_{(i,j)_k} = [v_i^{(t)}, v_j^{(t)}, u_{(i,j)_k}]$
        total_nbr_fea = torch.cat(
            [
                # (1) atom_in_fea: (N, F)
                #     Each row is the feature vector $v_i^{(t)}$ for one "center" atom i.
                #     unsqueeze(1) -> (N, 1, F): add neighbor dimension.
                #     expand(N, M, F): duplicate $v_i^{(t)}$ for all M neighbors.
                #     Physically: every bond $(i,j)_k$ needs to know which atom i it originates from,
                #     so we attach the same $v_i^{(t)}$ to all its neighbor edges.
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                # (2) atom_nbr_fea: (N, M, F)
                #     Neighbor atom features $v_j^{(t)}$ for each neighbor j of atom i.
                atom_nbr_fea,
                # (3) nbr_fea: (N, M, B)
                #     Bond (edge) features $u_{(i,j)_k}$ that describe the bond between i and j
                #     (e.g., distance, direction, coordination geometry).
                nbr_fea,
            ],
            dim=2,  # concatenate along the feature axis → join features, not neighbors or atoms
            # after concatenation: total_nbr_fea has shape (N, M, 2F + B)
            # each edge feature vector now contains [center atom, neighbor atom, bond] info
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
        n_targets=1,
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        classification: bool
          Whether to perform classification instead of regression
        n_targets: int
          Number of regression targets (ignored for classification)
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.n_targets = n_targets
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)
            ])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, self.n_targets)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert (
            sum([len(idx_map) for idx_map in crystal_atom_idx])
            == atom_fea.data.shape[0]
        )
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)
