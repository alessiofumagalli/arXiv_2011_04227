import numpy as np
import porepy as pp

from multilayer_grid import multilayer_grid_bucket

# ------------------------------------------------------------------------------#

def create_gb(mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    file_name = "network.csv"
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # assign the flag for the low permeable fractures
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # Generate a mixed-dimensional mesh
    gb = network.mesh(mesh_kwargs)

    # construct the multi-layer grid bucket, we give also a name to the fault and layer grids
    gb_multilayer = multilayer_grid_bucket(gb)
    #pp.plot_grid(gb_multilayer, alpha=0, info="cf")

    for _, d in gb_multilayer:
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}

    for _, d in gb_multilayer.edges():
        d[pp.STATE] = {}
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.COUPLING_DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}

    return gb_multilayer

# ------------------------------------------------------------------------------#

def get_param():
    # data problem

    tol = 1e-6
    end_time = 0.20
    num_steps = 100
    time_step = end_time / float(num_steps)

    return {
        "tol": tol,

        "time": {
            "end_time": end_time,
            "num_steps": num_steps,
            "step": time_step
        },

        # porosity
        "porosity": {
            "eta": 5*1e-2,
            "initial": initial_porosity
        },

        # fracture aperture
        "fracture_aperture": {
            "eta": 5*1e-2,
            "initial": initial_fracture_aperture
        },

        # layer aperture
        "layer_aperture": {
            "cutoff": 1e-1,
            "initial": initial_layer_aperture
        },

        # layer porosity
        "layer_porosity": {
            "eta": 5*1e-2,
            "initial": initial_layer_porosity
        },

        # flow
        "flow": {
            "tol": tol,
            "k_n": k_n,
            "k_t": k_t,

            "bc": bc_flow,
        },

        # temperature
        "temperature": {
            "tol": tol,
            "l_w": 1,
            "l_s": 1e-1,
            "rc_w": 1.0, # rho_w \cdot c_w
            "rc_s": 1.0,

            "bc": bc_temperature,
            "initial": initial_temperature,
        },

        # advection and diffusion of solute
        "solute_advection_diffusion": {
            "tol": tol,
            "d": 1e-8,
            "fracture_d_t": 1e-6, "fracture_d_n": 1e-6,
            "layer_d_t": 1e-6, "layer_d_n": 1e-6,

            "bc": bc_solute,
            "initial_solute": initial_solute,
            "initial_precipitate": initial_precipitate,
        },

        # reaction of solute and precipitate
        "solute_precipitate_reaction": {
            "tol": tol,
            "length": 1,
            "velocity": 1,
            "gamma_eq": 1,
            "theta": 0,
            "reaction": reaction_fct,
            "lambda": lambda_fct,
            "tol_reaction": 1e-12,
            "tol_consider_zero": 1e-30,
            "max_iter": 1e2,
        },

    }

# ------------------------------------------------------------------------------#

# reaction function, parameter: solute (u), precipitate (w), temperature (theta)
def reaction_fct(u, w, theta, tol=1e-15):
    l = lambda_fct(theta)
    r = np.power(u, 2)
    return -l*u
    return l*((w>tol)*np.maximum(1 - r, 0) - np.maximum(r - 1, 0))

# ------------------------------------------------------------------------------#

def lambda_fct(theta):
    return 100 #10*np.exp(-4/theta)

# ------------------------------------------------------------------------------#

def bc_flow(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 1 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[in_flow] = "dir"
    labels[out_flow] = "dir"
    bc_val[b_faces[in_flow]] = 1e-1
    bc_val[b_faces[out_flow]] = 0

    return labels, bc_val

# ------------------------------------------------------------------------------#

def bc_temperature(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 1 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels_diff = np.array(["neu"] * b_faces.size)
    labels_adv = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels_diff[in_flow] = "dir"
    labels_adv[in_flow] = "dir"
    labels_adv[out_flow] = "dir"

    bc_val[b_faces[in_flow]] = 1.5
    bc_val[b_faces[out_flow]] = 0

    return labels_diff, labels_adv, bc_val

# ------------------------------------------------------------------------------#

def bc_solute(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 1 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels_diff = np.array(["neu"] * b_faces.size)
    labels_adv = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels_diff[in_flow] = "dir"
    labels_adv[in_flow] = "dir"
    labels_adv[out_flow] = "dir"

    bc_val[b_faces[in_flow]] = 2
    bc_val[b_faces[out_flow]] = 0

    return labels_diff, labels_adv, bc_val

# ------------------------------------------------------------------------------#

def k_n(g):
    if "fracture" in g.name:
        return 1e2 * np.ones(g.num_cells)
    elif "layer" in g.name:
        return np.ones(g.num_cells)
    else:
        raise ValueError

# ------------------------------------------------------------------------------#

def k_t(g):
    if "fracture" in g.name:
        return 1e2 * np.ones(g.num_cells)
    elif "layer" in g.name:
        return np.ones(g.num_cells)
    else:
        return np.ones(g.num_cells)

# ------------------------------------------------------------------------------#

def initial_temperature(g, data, tol):
    temperature = np.ones(g.num_cells)
    return temperature

# ------------------------------------------------------------------------------#

def initial_solute(g, data, tol):
    solute = 0 * np.ones(g.num_cells)
    return solute

# ------------------------------------------------------------------------------#

def initial_precipitate(g, data, tol):
    precipitate = 0 * np.ones(g.num_cells)
    return precipitate

# ------------------------------------------------------------------------------#

def initial_porosity(g, data, tol):
    if g.dim == 2: # NOTE: we are in 2d as dim_max
        return 0.2 * np.ones(g.num_cells)
    else:
        # we set a zero porosity, meaning it is not active anymore
        # the temporal scheme considered keeps this variable null
        return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#

def initial_layer_porosity(g, data, tol):
    if "layer" in g.name:
        return 0.2 * np.ones(g.num_cells)
    else:
        return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#

def initial_fracture_aperture(g, data, tol):
    if "fracture" in g.name:
        return 1e-3*np.ones(g.num_cells)
    else:
        # we set a zero aperture, meaning it is not active
        # the temporal scheme considered keeps this variable null
        return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#

def initial_layer_aperture(g, data, tol):
    if "layer" in g.name:
        return 1e-8 * np.ones(g.num_cells)
    else:
        # we set a zero aperture, meaning it is not active
        # the temporal scheme considered keeps this variable null
        return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#
