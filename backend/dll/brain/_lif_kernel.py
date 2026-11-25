"""Numba kernel for LIFBrain: exact lazy ("event-driven") integration.

Same model and per-step order as LIFBrain._step_numpy (state update -> threshold ->
delayed synaptic arrivals, dropped for refractory targets -> Poisson input -> reset),
but most neurons are not integrated step by step.

Between inputs the membrane obeys a linear ODE with closed-form solution. Writing
u = v - v_0 and a = 1/t_mbr, b = 1/tau (b > a), after s seconds without input:

    u(s) = u0 * e^{-a s} + C * (e^{-a s} - e^{-b s}),   C = g0 * tau / (t_mbr - tau)

Because 0 <= e^{-a s} - e^{-b s} <= M (its maximum over s >= 0), we get the exact bound

    u(s) <= max(u0, 0) + max(C, 0) * M     for every s >= 0.

A neuron whose bound is below the threshold gap (v_th - v_0) cannot spike until it
receives new input, so it is left "cold": its state is advanced in closed form only
when an input arrives (or at the end of the call). Only "hot" neurons, whose bound
reaches the threshold, are integrated every step, exactly as in the dense model.
No rest-snapping approximation is involved; results differ from dense stepping only
by floating-point rounding (closed-form powers vs repeated multiplication).

Pending synaptic input per delay slot is tracked with a touched-index list, so no
step scans all N neurons. Returns True if the spike record buffer overflowed.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def lif_steps(
    n_steps, t0, delay_steps, rfc_steps,
    v_0, v_rst, v_th, decay_v, decay_g, g_to_v, poisson_w, std_u, dt_over_tau_rec,
    dv_tab, dg_tab, gv_tab, dt_over_tm, dt_over_tau, gv_coef, c_coef, m_peak,
    indptr, indices, w, input_idx, events, is_input,
    v, g, ref_until, buffer, bt_idx, bt_n, bt_flag, active, spike_count, std_x, std_last,
    spk_out, spk_ptr,
):
    n = v.shape[0]
    gap = v_th - v_0 - 1e-12
    n_tab = dv_tab.shape[0]
    t_upd = np.full(n, t0 - 1, dtype=np.int64)  # state is current at step t0 - 1
    in_hot = np.zeros(n, dtype=np.bool_)
    hot = np.empty(n, dtype=np.int64)
    n_hot = 0
    for i in range(n):
        u = v[i] - v_0
        if u != 0.0 or g[i] != 0.0:
            if max(u, 0.0) + max(g[i] * c_coef, 0.0) * m_peak >= gap:
                in_hot[i] = True
                hot[n_hot] = i
                n_hot += 1
    spk = np.empty(n, dtype=np.int64)
    n_out = 0
    truncated = False
    for k in range(n_steps):
        t = t0 + k
        slot = t % delay_steps
        # 1+2. integrate hot neurons one step (they are current at t-1), detect threshold
        n_spk = 0
        for q in range(n_hot):
            i = hot[q]
            if ref_until[i] <= t:
                gi = g[i]
                v[i] = v_0 + (v[i] - v_0) * decay_v + gi * g_to_v
                g[i] = gi * decay_g
                if v[i] > v_th:
                    spk[n_spk] = i
                    n_spk += 1
            t_upd[i] = t
        # 3a. delayed synaptic arrivals
        for q in range(bt_n[slot]):
            j = bt_idx[slot, q]
            arrived = buffer[slot, j]
            buffer[slot, j] = 0.0
            bt_flag[slot, j] = False
            if ref_until[j] > t:
                continue  # Brian2 drops input to refractory `(unless refractory)` variables
            if not in_hot[j]:
                d = t - t_upd[j]
                t_upd[j] = t
                gj = g[j]
                uj = v[j] - v_0
                if d > 0 and (uj != 0.0 or gj != 0.0):
                    if d < n_tab:
                        v[j] = v_0 + uj * dv_tab[d] + gj * gv_tab[d]
                        g[j] = gj * dg_tab[d]
                    else:
                        ev = np.exp(-d * dt_over_tm)
                        eg = np.exp(-d * dt_over_tau)
                        v[j] = v_0 + uj * ev + gj * gv_coef * (eg - ev)
                        g[j] = gj * eg
            g[j] += arrived
            if not in_hot[j]:
                if max(v[j] - v_0, 0.0) + max(g[j] * c_coef, 0.0) * m_peak >= gap:
                    in_hot[j] = True
                    hot[n_hot] = j
                    n_hot += 1
        bt_n[slot] = 0
        # 3b. Poisson drive (one event always crosses threshold)
        for q in range(input_idx.shape[0]):
            e = events[k, q]
            if e > 0:
                j = input_idx[q]
                if not in_hot[j]:
                    d = t - t_upd[j]
                    t_upd[j] = t
                    gj = g[j]
                    uj = v[j] - v_0
                    if d > 0 and (uj != 0.0 or gj != 0.0):
                        if d < n_tab:
                            v[j] = v_0 + uj * dv_tab[d] + gj * gv_tab[d]
                            g[j] = gj * dg_tab[d]
                        else:
                            ev = np.exp(-d * dt_over_tm)
                            eg = np.exp(-d * dt_over_tau)
                            v[j] = v_0 + uj * ev + gj * gv_coef * (eg - ev)
                            g[j] = gj * eg
                    in_hot[j] = True
                    hot[n_hot] = j
                    n_hot += 1
                v[j] += e * poisson_w
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
        # 5. neurons that can no longer reach threshold become cold (state current at t)
        q = 0
        while q < n_hot:
            i = hot[q]
            if max(v[i] - v_0, 0.0) + max(g[i] * c_coef, 0.0) * m_peak < gap:
                in_hot[i] = False
                n_hot -= 1
                hot[q] = hot[n_hot]
            else:
                q += 1
    # bring every neuron's state up to the last step so callers see current values
    t_last = t0 + n_steps - 1
    for i in range(n):
        gi = g[i]
        ui = v[i] - v_0
        nonrest = ui != 0.0 or gi != 0.0
        d = t_last - t_upd[i]
        if nonrest and d > 0:
            if d < n_tab:
                v[i] = v_0 + ui * dv_tab[d] + gi * gv_tab[d]
                g[i] = gi * dg_tab[d]
            else:
                ev = np.exp(-d * dt_over_tm)
                eg = np.exp(-d * dt_over_tau)
                v[i] = v_0 + ui * ev + gi * gv_coef * (eg - ev)
                g[i] = gi * eg
        active[i] = nonrest
    return truncated
