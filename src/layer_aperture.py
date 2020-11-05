import numpy as np
import porepy as pp

class LayerAperture(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="layer_aperture"):
        # NOTE: the structure of the dofs are inherited from the solute transport equation
        # set the discretization for the grids

        self.model = model
        self.gb = gb
        self.data = None
        self.data_time = None

        self.assembler = None

        self.variable = self.model + "_variable"

        # post process variables
        self.scalar = self.model + "_scalar"
        self.mortar_dummy_1 = self.model + "_lambda_dummy_1"
        self.mortar_dummy_2 = self.model + "_lambda_dummy_2"

        # set the discretizaton
        self.set_discr()

        self.is_set = False

    # ------------------------------------------------------------------------------#

    def set_discr(self):

        for _, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.variable: {"cells": 1}})

        for _, d in self.gb.edges():
            d[pp.PRIMARY_VARIABLES].update({self.mortar_dummy_1: {"cells": 1},
                                            self.mortar_dummy_2: {"cells": 1}})

        # assembler
        self.assembler = pp.Assembler(self.gb)

    # ------------------------------------------------------------------------------#

    def set_data(self, data, data_time):
        self.data = data
        self.data_time = data_time

    # ------------------------------------------------------------------------------#

    def shape(self):
        return self.gb.num_cells() + 2*self.gb.num_mortar_cells()

    # ------------------------------------------------------------------------------#

    def step(self, flux, porosity, lmbda, time, solute, aperture_min = 1e-12):

        if np.isscalar(lmbda):
            lmbda = lmbda * np.ones(flux.size)

        flux = np.clip(flux, 0., None)

        S = np.zeros(flux.size)
        mask = solute > 0
        S[mask] = -np.log(self.data["cutoff"] / solute[mask]) * flux[mask] / (porosity[mask] * lmbda[mask])

        threshold = time * flux / porosity
        mask = S >= threshold
        S[mask] = time * flux[mask] / porosity[mask]

        # the aperture cannot be zero put a small value
        S[S < aperture_min] = aperture_min
        S[np.logical_not(np.isfinite(S))] = aperture_min

        return S

    # ------------------------------------------------------------------------------#

    def extract(self, x, name=None):
        self.assembler.distribute_variable(x)
        if name is None:
            name = self.scalar
        for _, d in self.gb:
            d[pp.STATE][name] = d[pp.STATE][self.variable]

    # ------------------------------------------------------------------------------#
