import numpy as np

# Need to tune gain and reqrite in cpp
def apply_gardner_ted(signal: np.ndarray, sps: int, gain: float = 0.01) -> np.ndarray:
    def cubic_interp(s, idx, mu):
        if idx < 1 or idx + 2 >= len(s):
            return s[np.clip(idx, 0, len(s)-1)]
        v0, v1, v2, v3 = s[idx-1], s[idx], s[idx+1], s[idx+2]
        c0 =  v1
        c1 = -0.5*v0 + 0.5*v2
        c2 =  v0 - 2.5*v1 + 2.0*v2 - 0.5*v3
        c3 = -0.5*v0 + 1.5*v1 - 1.5*v2 + 0.5*v3
        return c0 + c1*mu + c2*mu**2 + c3*mu**3

    mu = 0.0  # fractional timing offset, persists across iterations
    out = []
    k = float(sps)

    while True:
        k_int     = int(k)
        k_mid     = k - sps / 2
        k_mid_int = int(k_mid)

        if k_int + 2 >= len(signal) or k_mid_int < 1:
            break

        y_curr = cubic_interp(signal, k_int,     k - k_int)
        y_prev = cubic_interp(signal, k_int - sps, k - k_int)
        y_mid  = cubic_interp(signal, k_mid_int,  k_mid - k_mid_int)  # correct fractional offset

        e  = np.real((y_curr - y_prev) * np.conj(y_mid))
        mu = np.clip(mu + gain * e, -0.5, 0.5)  # accumulate, don't reset
        #print(e)
        k += sps + mu  # apply accumulated correction

        out.append(y_curr)

    return np.array(out)

