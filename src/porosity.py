import porepy as pp

class Porosity(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="porosity"):
        # NOTE: the structure of the dofs are inherited from the solute transport equation
        # set the discretization for the grids

        self.model = model
        self.gb = gb
        self.data = None
        self.eta = None

        self.assembler = None

        self.variable = self.model + "_variable"

        # post process variables
        self.scalar = self.model + "_scalar"
        self.mortar_dummy_1 = self.model + "_lambda_dummy_1"
        self.mortar_dummy_2 = self.model + "_lambda_dummy_2"

        # set the discretizaton
        self.set_discr()

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

    def set_data(self, data):
        self.data = data
        self.eta = data["eta"]

    # ------------------------------------------------------------------------------#

    def shape(self):
        return self.gb.num_cells() + 2*self.gb.num_mortar_cells()

    # ------------------------------------------------------------------------------#

    def step(self, porosity_old, precipitate, precipitate_old):
        return porosity_old / (1 + self.eta*(precipitate - precipitate_old))

    # ------------------------------------------------------------------------------#

    def extract(self, x, name=None):
        self.assembler.distribute_variable(x)
        if name is None:
            name = self.scalar
        for _, d in self.gb:
            d[pp.STATE][name] = d[pp.STATE][self.variable]

    # ------------------------------------------------------------------------------#
