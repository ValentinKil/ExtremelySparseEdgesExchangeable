import numpy as np
from scipy.special import gamma

def rapidbeta(eta, xi, num_components=None, truncation_level=1000):
    """
    Samples the jump sizes (weights) W of a Completely Random Measure (CRM)
    by representing it as a mixture of Generalized Gamma Processes (GGPs).

    This method is based on the unified framework for CRMs and the novel GGP
    series representation described in "A unified construction for series
    representations and finite approximations of completely random measures"
    (Lee, Miscouridou, Caron, 2019).

    Args:
        eta (float): The eta (η) parameter of the target CRM's Lévy intensity.
                     Controls the overall number of jumps. Must be > 0.
        xi (float): The xi (ξ) parameter of the target CRM's Lévy intensity.
                    Controls the tail behavior. Must be > 1 for this method.
        num_components (int, optional): The number of GGP components (J) in the
                                        mixture. If None, it is sampled from a
                                        Poisson(eta / 2) distribution.
                                        Defaults to None.
        truncation_level (int, optional): The number of jumps to simulate for
                                          each GGP component in the mixture.
                                          Defaults to 1000.

    Returns:
        np.array: An array of the simulated jump sizes W, sorted in
                  descending order.
    """
    if not eta > 0:
        raise ValueError("Parameter 'eta' must be positive.")
    if not xi > 1:
        raise ValueError("Parameter 'xi' must be greater than 1.")

    # --- Step 1: Sample the number of GGP components (J) ---
    # The number of components in the mixture is a Poisson random variable.
    if num_components is None:
        J = np.random.poisson(eta / 2)
    else:
        J = num_components
        
    if J == 0:
        print("Warning: Sampled 0 GGP components. Returning empty array.")
        return np.array([])

    all_v_jumps = []

    # --- Step 2 & 3: For each component, sample parameters and GGP jumps ---
    for j in range(J):
        # --- Step 2A: Sample GGP parameters (S_j, T_j) ---
        # Sample the stability parameter S_j from p(s) = 2s for s in (0,1)
        # using inverse transform sampling. CDF is F(s) = s^2, inverse is sqrt(u).
        u_sample = np.random.uniform(0, 1)
        S_j = np.sqrt(u_sample)

        # Sample the tempering parameter T_j from Gamma(xi - S_j, 1)
        T_j = np.random.gamma(shape=xi - S_j, scale=1.0)

        # --- Step 3A: Simulate jumps V_jk from the GGP(S_j, T_j) ---
        # This uses the novel sequential construction from the paper.

        # Generate event times of a unit-rate Poisson process
        # by taking the cumulative sum of unit-rate exponential variables.
        exp_samples = np.random.exponential(scale=1.0, size=truncation_level)
        xi_jk = np.cumsum(exp_samples)

        # GGP parameters for the simulation formula
        sigma = S_j
        tau = T_j
        # The scale parameter alpha is derived from the mixing intensity
        alpha = gamma(1 - sigma)

        # --- Step 3B: Calculate GGP jump sizes V_jk ---
        # The formula for the scale of the conditional Gamma distribution is:
        # ( (sigma * xi_jk / alpha) + tau**sigma )**(1/sigma)
        gamma_scale = ( (sigma * xi_jk / alpha) + (tau**sigma) )**(1 / sigma)
        
        # Sample the jumps V_jk from the conditional Gamma distribution
        # The shape is (1 - sigma).
        v_jumps = np.random.gamma(shape=1 - sigma, scale=gamma_scale)
        all_v_jumps.append(v_jumps)

    # --- Step 4: Combine and Transform Jumps ---
    # Combine all jumps from all components into a single array
    V = np.concatenate(all_v_jumps)

    # Transform the V jumps (from v-space) back to W jumps (w-space)
    W = V / (1.0 + V)

    return W