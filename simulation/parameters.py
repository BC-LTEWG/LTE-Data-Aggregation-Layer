from dataclasses import dataclass, field
from numpy import array, ndarray


@dataclass
class Params:
    N_c: int = 5
    N_S: int = 10000
    N_h: int = 100
    m_r: int = 5
    N_p: int = 5
    N_d: int = 5
    W: int = 5
    T: int = 8
    S: float = 0.01
    v_ability: float = 0.0005
    N_a: int = 5
    is_logging: bool = False
