from dataclasses import dataclass


@dataclass
class TwoModeSetting:
    X_MIN, X_MAX, N_X = -8, 8, 500
    Y_MAX, INTERVAL = 0.75, 100
    TEXT_POSITION_T = (0.29, 0.5)
    TEXT_POSITION_T_PRIME = (0.69, 0.5)

    NUM_ITERS = 200
    NUM_SAMPLES = 1000
    FWD_LR = [
        0.15,
        0.01,
    ]
    REV_LR = [
        0.01,
    ]
    REINFORCE_BETA = 0.1
    SLOW_FACTOR = 0.005
   
    # q(x) = pi1 * N(mu1, sig1) + (1-pi1) * N(mu2, sig2)
    PI1 = 0.75
    MU1, SIG1 = -3.5, 1.0
    MU2, SIG2 = 0.5, 0.7
    
    # p(x) = pi1 * N(mu1, sig1) + (1-pi1) * N(mu2, sig2)
    MIXTURE = [
        (0.75, -3.0, 1.0),
        (0.25, 3.5, 0.7),
    ]


@dataclass
class SingleModeSetting:
    X_MIN, X_MAX, N_X = -10, 10, 500
    Y_MAX, INTERVAL = 0.75, 100
    TEXT_POSITION_T = (0.29, 0.5)
    TEXT_POSITION_T_PRIME = (0.69, 0.5)

    #NUM_ITERS = 200
    NUM_ITERS = 1000
    NUM_SAMPLES = 1000
    FWD_LR = [
        0.05, # 90, 64
    ]
    REV_LR = [
        0.005, # 90, 70
    ]
    REINFORCE_BETA = 0.1
   
    # q(x) = N(mu1, sig1)
    PI1 = 1.0
    MU1, SIG1 = -3.5, 1.0
    
    # p(x) = pi1 * N(mu1, sig1) + (1-pi1) * N(mu2, sig2)
    MIXTURE = [
        (0.75, -3.0, 1.0),
        (0.25, 3.5, 0.7),
    ]
