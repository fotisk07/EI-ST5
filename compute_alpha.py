from math import exp


def compute_alpha(omega):
    return (5.5 + (15 - 5.5)*(1 - exp(-omega/5000))) - 1j*omega/300
