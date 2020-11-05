# A multi-layer reactive transport model for fractured porous media

Source code and examples for the paper<br>
"*A multi-layer reactive transport model for fractured porous media*" by Luca Formaggia, Alessio Fumagalli, Anna Scotti. See [arXiv pre-print](https://arxiv.org/abs/**).


# Reproduce results from paper
Runscripts for all test cases of the work available [here](./examples).<br>
Note that you may have to revert to an older version of [PorePy](https://github.com/pmgbergen/porepy) to run the examples.

# Abstract
The accurate modeling of reactive flows in fractured porous media is a key ingredient to
obtain reliable numerical simulations of several industrial and environmental applications. For some
values of the physical parameters we can observe the formation of a narrow region or layer around
the fractures where chemical reactions are focused. Here the transported solute may precipitate and
form a salt, or vice-versa. This phenomenon has been observed and reported in real outcrops. By
changing its physical properties this layer might substantially alter the global flow response of the
system and thus the actual transport of solute: the problem is thus non-linear and fully coupled. The
aim of this work is to propose a new mathematical model for reactive flow in fractured porous media,
by approximating both the fracture and these surrounding layers via a reduced model. In particular, our
main goal is to describe the layer thickness evolution with a new mathematical model, and compare
it to a fully resolved equidimensional model for validation. As concerns numerical approximation
we extend an operator splitting scheme in time to solve sequentially, at each time step, each physical
process thus avoiding the need for a non-linear monolithic solver, which might be challenging due to
the non-smoothness of the reaction rate. We consider bi- and tridimensional numerical test cases to
asses the accuracy and benefit of the proposed model in realistic scenarios.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv:** [math.NA]](https://arxiv.org/abs/**).

# PorePy version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and revert to ** <br>
Newer versions of PorePy may not be compatible with this repository.

# License
See [license](./LICENSE).
