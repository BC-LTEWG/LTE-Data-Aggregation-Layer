from dataclasses import dataclass, field
from numpy import array, ndarray


@dataclass
class Params:
    N_c: int = 5
    N_S: int = 10000
    N_h: int = 100
    # m_r: int = 5
    N_inputs_max: int = 5
    N_m: int = 1
    N_p: int = 5
    N_d: int = 5
    W: int = 5
    T: int = 8
    S: float = 0.1
    v_ability: float = 0.1
    N_a: int = 5
    fixed_seed: bool = False
    seed: int = 0
    is_logging: bool = False
    productivity: float = 0.5
    consump_epsilon: float = 0.6
    init_prices: str = 'labor_values'
