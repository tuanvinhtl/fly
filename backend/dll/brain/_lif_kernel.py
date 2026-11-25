"""Numba kernel (returns True if the spike record buffer overflowed) for LIFBrain — same semantics as LIFBrain._step_numpy (tested for equality).

Active neurons are tracked as a compact index list; pending synaptic input per delay
slot is tracked with a touched-index list so no step scans all N neurons.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def lif_steps(
    n_steps, t0, delay_steps, rfc_steps,
    v_0, v_rst, v_th, decay_v, decay_g, g_to_v, poisson_w, rest_tol, std_u, dt_over_tau_rec,
    indptr, indices, w, input_idx, events, is_input,
    v, g, ref_until, buffer, bt_idx, bt_n, bt_flag, active, spike_count, std_x, std_last,
    spk_out, spk_ptr,
):
    n = v.shape[0]
    use_active = rest_tol > 0.0
    act = np.empty(n, dtype=np.int64)
    n_act = 0
    if use_active:
        for i in range(n):
            if active[i]:
                act[n_act] = i
                n_act += 1
    else:
        for i in range(n):
            act[i] = i
            active[i] = True
        n_act = n
    spk = np.empty(n, dtype=np.int64)
    n_out = 0
    truncated = False
    for k in range(n_steps):
        t = t0 + k
        slot = t % delay_steps
        # 1+2. integrate non-refractory active neurons, detect threshold crossings
        n_spk = 0
        for q in range(n_act):
            i = act[q]
            if ref_until[i] <= t:
                gi = g[i]
                v[i] = v_0 + (v[i] - v_0) * decay_v + gi * g_to_v
                g[i] = gi * decay_g
                if v[i] > v_th:
                    spk[n_spk] = i
                    n_spk += 1
        # 3a. delayed synaptic arrivals
        for q in range(bt_n[slot]):
            j = bt_idx[slot, q]
            arrived = buffer[slot, j]
            buffer[slot, j] = 0.0
            bt_flag[slot, j] = False
            if ref_until[j] > t:
                continue  # Brian2 drops input to refractory `(unless refractory)` variables
            g[j] += arrived
            if not active[j]:
                active[j] = True
                act[n_act] = j
                n_act += 1
        bt_n[slot] = 0
        # 3b. Poisson drive
        for q in range(input_idx.shape[0]):
            e = events[k, q]
            if e > 0:
                j = input_idx[q]
                v[j] += e * poisson_w
                if not active[j]:
                    active[j] = True
                    act[n_act] = j
                    n_act += 1
        # 3c. outgoing spikes + 4. reset
        for s in range(n_spk):
            i = spk[s]
            eff = 1.0
            if std_u[i] > 0.0:
                eff = 1.0 - (1.0 - std_x[i]) * np.exp(-(t - std_last[i]) * dt_over_tau_rec)
                std_x[i] = eff * (1.0 - std_u[i])
                std_last[i] = t
            for e in range(indptr[i], indptr[i + 1]):
                j = indices[e]
                wij = w[e] * eff
                if wij != 0.0:
                    buffer[slot, j] += wij
                    if not bt_flag[slot, j]:
                        bt_flag[slot, j] = True
                        bt_idx[slot, bt_n[slot]] = j
                        bt_n[slot] += 1
            v[i] = v_rst
            g[i] = 0.0
            ref_until[i] = t + (0 if is_input[i] else rfc_steps)
            spike_count[i] += 1
            if n_out < spk_out.shape[0]:
                spk_out[n_out] = i
                n_out += 1
            else:
                truncated = True
        spk_ptr[k + 1] = n_out
        # 5. drop neurons that returned to rest
        if use_active:
            q = 0
            while q < n_act:
                i = act[q]
                if abs(v[i] - v_0) < rest_tol and abs(g[i]) < rest_tol and ref_until[i] <= t + 1:
                    v[i] = v_0
                    g[i] = 0.0
                    active[i] = False
                    n_act -= 1
                    act[q] = act[n_act]
                else:
                    q += 1
    return truncated
