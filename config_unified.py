

import numpy as np


class GlobalConfig:
    random_seed = 42
    verbose = True


class DatasetConfig:
    dataset = 'diffdrive'
    normalize = True
    normalize_range = (-1.0, 1.0)


class MethodConfig:
    method_names = ('RC_Koopman', 'EDMD', 'HAVOK')
    reconstruction_method = 'one-step-ahead'


class DuffingConfig:
    alpha = 1.0
    beta = -1.0
    gamma = 0.5
    t_start = 0.0
    t_end = 20.0
    time_step = 0.04
    initial_condition = np.array([-1.21, 0.81])


class DiffDriveConfig:
    dt = 0.05
    n_trajectories = 200
    steps_per_traj = 50
    x_range = (-2.0, 2.0)
    y_range = (-2.0, 2.0)
    theta_range = (-np.pi, np.pi)
    v_range = (-0.6, 0.6)
    omega_range = (-2.0, 2.0)


class RCKoopmanConfig:
    lifted_dim_D = 12
    rho_wres = 0.9
    w_range = 1.0
    win_range = 1.0
    density = 1
    activation = 'tanh'
    washout = 20
    noise_level = 0.0
    use_regularization = False
    regularization_param = 1e-4
    rcond = 1e-10


class EDMDConfig:
    lifted_dim_D = 11
    poly_order = 1
    use_trig = True
    use_cross_terms = True
    use_rbf = False
    rbf_eta = 120
    rbf_center_method = 'random'
    include_constant = False
    include_u_in_phi = False
    use_regularization = False
    regularization_param = 1e-6
    rcond = 1e-15


class HAVOKConfig:
    embedding_dim = 6
    delay_steps = 1
    include_input_in_hankel = True
    use_regularization = False
    regularization_param = 1e-6
    rcond = 1e-15
