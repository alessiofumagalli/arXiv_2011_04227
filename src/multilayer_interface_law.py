import numpy as np
import scipy.sparse as sps

import porepy as pp

class RobinCouplingMultilayer(pp.RobinCoupling):
    """ A condition with resistance to flow between subdomains. Implementation
        of the model studied (though not originally proposed) by Martin et
        al 2005.

        @ALL: We should probably make an abstract superclass for all couplers,
        similar to for all elliptic discretizations, so that new
        implementations know what must be done.

    """
    def __init__(self, keyword, discr_master, discr_slave=None) -> None:
        super(RobinCouplingMultilayer, self).__init__(keyword, discr_master, discr_slave)
        self.edge_coupling_via_high_dim = None
        self.edge_coupling_via_low_dim = None

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """ Discretize the interface law and store the discretization in the
        edge data.

        Parameters:
            g_h: Grid of the master domanin.
            g_l: Grid of the slave domain.
            data_h: Data dictionary for the master domain.
            data_l: Data dictionary for the slave domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        parameter_dictionary_h = data_h[pp.PARAMETERS][self.discr_master.keyword]
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        kn = parameter_dictionary_edge["normal_diffusivity"]
        # If normal diffusivity is given as a constant, parse to np.array
        if not isinstance(kn, np.ndarray):
            kn *= np.ones(mg.num_cells)

        inv_M = sps.diags(1.0 / mg.cell_volumes)
        inv_k = 1.0 / kn
        Eta = sps.diags(inv_k)
        matrix_dictionary_edge[self.mortar_discr_key] = -inv_M * Eta

        if self.kinv_scaling:
            # Use a discretization fit for mixed methods, with a K^-1 scaling of the
            # mortar flux
            # In this case, the scaling of the pressure blocks on the mortar rows is
            # simple.
            matrix_dictionary_edge[self.mortar_scaling_key] = sps.diags(
                np.ones(mg.num_cells)
            )

        else:
            # Scale the the mortar equations with K, so that the this becomes a
            # Darcy-type equation on standard form.
            matrix_dictionary_edge[self.mortar_scaling_key] = sps.diags(
                mg.cell_volumes * kn
            )

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):
        """ Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            g_master: Grid on one neighboring subdomain.
            g_slave: Grid on the other neighboring subdomain.
            data_master: Data dictionary for the master suddomain
            data_slave: Data dictionary for the slave subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_master: original discretization for the master subdomain
            matrix_slave: original discretization for the slave subdomain

            The discretization matrices must be included since they will be
            changed by the imposition of Neumann boundary conditions on the
            internal boundary in some numerical methods (Read: VEM, RT0)

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        mg = data_edge["mortar_grid"]

        master_ind = 0
        slave_ind = 1
        cc, rhs = self._define_local_block_matrix(
            g_master, g_slave, self.discr_master, self.discr_slave, mg, matrix
        )

        # The convention, for now, is to put the higher dimensional information
        # in the first column and row in matrix, lower-dimensional in the second
        # and mortar variables in the third
        cc[2, 2] = matrix_dictionary_edge[self.mortar_discr_key]

        self.discr_master.assemble_int_bound_pressure_cell_1(
            g_master, data_master, data_edge, True, cc, matrix, rhs, master_ind, coeff = -1
        )
        self.discr_master.assemble_int_bound_source_1(
            g_master, data_master, data_edge, True, cc, matrix, rhs, master_ind, coeff = -1
        )

        self.discr_slave.assemble_int_bound_pressure_cell_1(
            g_slave, data_slave, data_edge, False, cc, matrix, rhs, slave_ind, coeff = 1
        )
        self.discr_slave.assemble_int_bound_source_1(
            g_slave, data_slave, data_edge, False, cc, matrix, rhs, slave_ind, coeff = 1
        )

        matrix += cc

        return matrix, rhs

# ------------------------------------------------------------------------------
