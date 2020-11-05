import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#

def create_gb(mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

    delta = 0.00125
    frac_pts = np.array([[0.1, 0.9],
                         [0, 0.8]])

    norm = np.array([[-frac_pts[1, 1]+frac_pts[1, 0], frac_pts[0, 1]-frac_pts[0, 0]]])
    norm /= np.linalg.norm(norm)

    pts = np.hstack((frac_pts, frac_pts + delta*norm.T, frac_pts - delta*norm.T))

    edges = np.array([[0, 2, 4],
                      [1, 3, 5]])

    # assign the flag for the low permeable fractures
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 200}

    # Generate a mixed-dimensional mesh
    network = pp.FractureNetwork2d(pts, edges, domain)

    # Generate a mixed-dimensional mesh
    gb = network.mesh(mesh_kwargs, constraints=[1, 2])

    for _, d in gb.edges():
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.COUPLING_DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}
        d[pp.STATE] = {}

    # identification of layer and fracture
    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}
        d[pp.STATE] = {}

        if g.dim < gb.dim_max():
            g.name += ["fracture"]

        # save the identification of the fracture
        if "fracture" in g.name:
            d[pp.STATE]["fracture"] = np.ones(g.num_cells)
            d[pp.STATE]["layer"] = np.zeros(g.num_cells)
        # save zero for the other cases
        else:
            d[pp.STATE]["fracture"] = np.zeros(g.num_cells)
            d[pp.STATE]["layer"] = np.zeros(g.num_cells)

    return gb

# ------------------------------------------------------------------------------#

def get_param():
    # data problem

    tol = 1e-6
    end_time = 0.2
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
            "eta": 0,
            "initial": initial_porosity
        },

        # fracture aperture
        "fracture_aperture": {
            "eta": 0,
            "initial": initial_fracture_aperture
        },

        # layer aperture
        "layer_aperture": {
            "cutoff": 1e-1,
            "initial": initial_layer_aperture
        },

        # layer porosity
        "layer_porosity": {
            "eta": 0,
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

# ------------------------------------------------------------------------------#

def lambda_fct(theta):
    return 100

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
        return np.zeros(g.num_cells)
    else:
        raise ValueError

# ------------------------------------------------------------------------------#

def k_t(g):
    if "fracture" in g.name:
        return 1e2 * np.ones(g.num_cells)
    elif "layer" in g.name:
        return np.zeros(g.num_cells)
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
    return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#
