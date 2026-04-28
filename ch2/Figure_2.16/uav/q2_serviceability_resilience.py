import math


def q_function(x):
    """
    Gaussian Q-function:
    Q(x) = 0.5 * erfc(x / sqrt(2))
    """
    return 0.5 * math.erfc(x / math.sqrt(2))


def rms_micro_doppler_error(fs_hz, snr_db, return_loss_db):
    """
    Compute the RMS error of micro-Doppler frequency estimation
    based on Eq. (20) in Hou et al. (2021):

        10 log10 E(|epsilon_f|^2)
            < 20 log10(fs / pi) + RL - SNR

    Converting the bound into linear RMS form:

        sigma = sqrt(E(|epsilon_f|^2))
              = (fs / pi) * 10^((RL - SNR) / 20)

    Parameters
    ----------
    fs_hz : float
        Sampling frequency in Hz.
    snr_db : float
        Signal-to-noise ratio in dB.
    return_loss_db : float
        Return loss of the propeller reflection path in dB.

    Returns
    -------
    float
        RMS estimation error sigma in Hz.
    """
    sigma = (fs_hz / math.pi) * (10 ** ((return_loss_db - snr_db) / 20))
    return sigma


def serviceability_failure_probability(threshold_hz, sigma_hz, two_sided=True):
    """
    Compute the serviceability violation probability.

    The estimation error is modeled as:

        epsilon_f ~ N(0, sigma^2)

    If two_sided=True:
        P_fail = P(|epsilon_f| > T) = 2Q(T / sigma)

    If two_sided=False:
        P_fail = P(epsilon_f > T) = Q(T / sigma)

    In Q2, the two-sided version is used because both positive and negative
    estimation errors can violate the serviceability threshold.

    Parameters
    ----------
    threshold_hz : float
        Serviceability threshold T in Hz.
    sigma_hz : float
        RMS estimation error sigma in Hz.
    two_sided : bool
        Whether to calculate two-sided or one-sided violation probability.

    Returns
    -------
    float
        Serviceability violation probability.
    """
    z = threshold_hz / sigma_hz

    if two_sided:
        return 2 * q_function(z)
    return q_function(z)


def system_resilience_index(p_fail, n_uavs=10, k_required=7):
    """
    Compute the k-out-of-n system resilience index.

    The system is resilient if at least k_required UAVs remain serviceable.

    Let:
        P_success = 1 - P_fail
        X ~ Binomial(n_uavs, P_success)

    Then:
        R_sys = P(X >= k_required)
              = sum_{i=k}^{n} C(n, i) P_success^i P_fail^(n-i)

    Parameters
    ----------
    p_fail : float
        Single-UAV serviceability violation probability.
    n_uavs : int
        Total number of UAVs.
    k_required : int
        Minimum number of serviceable UAVs required for system resilience.

    Returns
    -------
    float
        System resilience index.
    """
    p_success = 1 - p_fail

    r_sys = 0.0
    for i in range(k_required, n_uavs + 1):
        r_sys += (
            math.comb(n_uavs, i)
            * (p_success ** i)
            * (p_fail ** (n_uavs - i))
        )

    return r_sys


if __name__ == "__main__":
    # Q2 given parameters
    SNR_DB = 10
    FS_HZ = 60_000
    RETURN_LOSS_DB = 5
    THRESHOLD_HZ = 50

    N_UAVS = 10
    K_REQUIRED = 7

    # Step 1: Compute RMS estimation error from Eq. (20)
    sigma = rms_micro_doppler_error(
        fs_hz=FS_HZ,
        snr_db=SNR_DB,
        return_loss_db=RETURN_LOSS_DB
    )

    # Step 2: Compute two-sided serviceability violation probability
    p_fail = serviceability_failure_probability(
        threshold_hz=THRESHOLD_HZ,
        sigma_hz=sigma,
        two_sided=True
    )

    p_success = 1 - p_fail

    # Optional: one-sided result if strictly following epsilon_f > T
    p_fail_one_sided = serviceability_failure_probability(
        threshold_hz=THRESHOLD_HZ,
        sigma_hz=sigma,
        two_sided=False
    )

    # Step 3: Compute k-out-of-n resilience index
    r_sys = system_resilience_index(
        p_fail=p_fail,
        n_uavs=N_UAVS,
        k_required=K_REQUIRED
    )

    r_sys_one_sided = system_resilience_index(
        p_fail=p_fail_one_sided,
        n_uavs=N_UAVS,
        k_required=K_REQUIRED
    )

    output = f"""=== Q2: Serviceability and Resilience Metrics ===
SNR = {SNR_DB} dB
Sampling rate fs = {FS_HZ} Hz
Return loss RL = {RETURN_LOSS_DB} dB
Serviceability threshold T = {THRESHOLD_HZ} Hz
Number of UAVs n = {N_UAVS}
Required serviceable UAVs k = {K_REQUIRED}

RMS estimation error sigma = {sigma:.4f} Hz

[Recommended two-sided serviceability violation]
P_fail = P(|epsilon| > T) = {p_fail:.6f}
P_success = {p_success:.6f}
R_sys = {r_sys:.6e}

[Optional one-sided result if using epsilon > T strictly]
P_fail_one_sided = P(epsilon > T) = {p_fail_one_sided:.6f}
R_sys_one_sided = {r_sys_one_sided:.6e}

"""

    print(output)

    with open("q2_serviceability_results.txt", "w", encoding="utf-8") as f:
        f.write(output)
