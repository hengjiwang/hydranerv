import numpy as np

def power_law_degree_pdf(k, alpha=2.5, k_min=1):
    """power law degree distribution"""
    return (alpha - 1) / k_min * (k / k_min) ** - alpha

def power_law_ccdf(k, alpha=2.5, k_min=1):
    """complementary cumulative distribution function"""
    return (k / k_min) ** - (alpha - 1)