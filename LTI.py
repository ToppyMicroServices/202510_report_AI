import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.linalg import eigvals, svd, inv, norm
from scipy.stats import pearsonr, spearmanr, linregress, kendalltau, theilslopes

# ==== EXPERIMENT CONFIG (control knobs) ====
CFG = {
    # Misspecification of measurement noise: v ~ N(0, R_true) but filter uses R_filt = scale * R_true
    'use_misspec': True,
    'R_filt_scale': 0.3,         # <1.0 makes the filter overconfident (hallucination-prone)

    # Q misspecification for process noise
    'use_Q_misspec': True,
    'Q_filt_scale': 0.12,   # <1.0 makes the filter underestimate process noise (overconfident dynamics)

    # Kalman gain policy across eps: 'dare_per_eps' or 'fixed_ref'
    'k_policy': 'fixed_ref',  # try 'fixed_ref' to disable per-eps re-tuning

    # NIS metrics
    'nis_use_window': True,
    'nis_window': 400,           # moving window length for local NIS mean
    'nis_quantile': 0.99,        # also report upper-quantile of Z^2

    # Validity cutoffs
    'rho_cutoff': 0.995,      # early-exit if spectral radius(A-KH) >= rho_cutoff
    'nis_max': 1e6,           # guard: if Z^2 or S explode beyond this, drop sample

    # Epsilon grid (customized sampling)
    'eps_mode': 'custom',
    # Focused dense sampling in range producing H_Risk ≈ 1--1.4
    'eps_custom': np.concatenate([
        np.linspace(0.25, 0.35, 15),   # dense central band (typical stable baseline)
        np.linspace(0.05, 0.25, 5),    # lower side (mild instability)
        np.logspace(-3, -1, 5)         # deeper near-degenerate region
    ]),
    # 'eps_min': 1e-6,
    # 'eps_max': 1e-1,
    # 'eps_points': 20,

    # Random seed
    'seed': 2025,

    # H scaling sweep (magnitude) and whether to scale R with s^2 (SNR-invariant vs SNR-changing)
    'H_scale_list': [1.0, 2.0, 5.0],
    'scale_R_with_H': False,   # if True, use R_true_eff = s^2 R_true and R_filt_eff = s^2 R_filt

    # Plot style (journal-ready)
    'plot_titles': False,     # caption-only policy: suppress in-figure titles
    'plot_fontsize': 10,
    'plot_dpi': 300,
    'run_supplemental_variants': True,
}

# ==== Parameters ====
A = np.array([[0.95, 0.60],[0.00, 0.97]])
sigma_w, sigma_v = 3e-2, 1e-2
Q_true = (sigma_w**2)*np.eye(2)
R_true = np.array([[sigma_v**2]])
# Filter-believed noises (misspec allowed)
Q_filt = (CFG['Q_filt_scale'] * (sigma_w**2))*np.eye(2) if CFG['use_Q_misspec'] else Q_true.copy()
R_filt = (CFG['R_filt_scale'] * sigma_v**2) * np.array([[1.0]]) if CFG['use_misspec'] else R_true.copy()
K0 = np.array([[0.28,0.12]])
alpha=1.0
K=alpha*K0
rng = np.random.default_rng(CFG['seed'])
# === Shared noise sequences for reproducibility ===
T_SIM = 20000
BURN = 2000
W_SEQ = rng.normal(0, np.sqrt(Q_true[0,0]), size=(T_SIM, 2, 1))
V_SEQ = rng.normal(0, np.sqrt(R_true[0,0]), size=(T_SIM, 1, 1))
plt.rcParams['font.size'] = CFG.get('plot_fontsize', 10)

# === Helper functions ===
def spectral_radius(M): return np.max(np.abs(eigvals(M)))
def cond_number(M): s=svd(M,compute_uv=False); return s[0]/s[-1]
def lyapunov_P(Phi,Sigma):
    I=np.eye(Phi.shape[0]**2); M=I-np.kron(Phi,Phi)
    vecP=np.linalg.pinv(M)@Sigma.reshape(-1,1)
    P=vecP.reshape(Phi.shape[0],Phi.shape[0]); return 0.5*(P+P.T),M
def integrated_sens(M): return norm(np.linalg.pinv(M),2)

# --- Helper for consistent axis labeling ---
def set_labels(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if CFG.get('plot_titles', True) and title is not None:
        ax.set_title(title)

# === Steady-state Kalman gain via fixed-point iteration (DARE) ===
def kalman_gain_ss(A, H, Q, R, tol=1e-10, maxit=10000):
    n = A.shape[0]
    P = np.copy(Q)  # warm start
    I = np.eye(n)
    for _ in range(maxit):
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P_new = A @ (P - K @ H @ P) @ A.T + Q
        if np.linalg.norm(P_new - P, ord='fro') < tol:
            P = 0.5*(P_new + P_new.T)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            return K, P
        P = P_new
    # final sync
    P = 0.5*(P + P.T)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return K, P

# === epsilon grid helper ===
def make_eps_grid(cfg):
    mode = cfg.get('eps_mode', 'log')
    if mode == 'linear':
        return np.linspace(cfg['eps_max'], cfg['eps_min'], cfg['eps_points'])
    elif mode == 'log':
        return np.concatenate([np.array([0.3]), np.logspace(np.log10(cfg['eps_max']), np.log10(cfg['eps_min']), cfg['eps_points'])])
    elif mode == 'custom' and 'eps_custom' in cfg:
        return np.unique(np.sort(cfg['eps_custom']))
    else:
        raise ValueError(f"Unsupported eps_mode: {mode}")

def effective_Rs(R_true, R_filt, s, scale_R_with_H):
    """Return (R_true_eff, R_filt_eff) depending on whether we keep SNR invariant when scaling H by s."""
    if scale_R_with_H:
        return ( (s**2) * R_true, (s**2) * R_filt )
    else:
        return ( R_true, R_filt )

# === Predictive-Z simulation with controllable misspec and windowed NIS ===
def simulate(A,H,Q_true,R_true,R_filt,Q_filt,K,T=20000,burn=2000,zstar=1.2,gamma_q=40, cfg=CFG, w_seq=None, v_seq=None):
    n=2; x=np.zeros((n,1)); xhat=np.zeros((n,1)); P=np.zeros((n,n))
    # Early exit if closed-loop is unstable: metrics undefined/meaningless
    Phi_chk = A - K @ H
    if spectral_radius(Phi_chk) >= cfg.get('rho_cutoff', 0.999):
        return np.nan, np.nan, np.nan
    S_hist, z2_hist = [], []
    for t in range(T):
        w = w_seq[t] if w_seq is not None else rng.normal(0, np.sqrt(Q_true[0,0]), size=(n,1))
        v = v_seq[t] if v_seq is not None else rng.normal(0, np.sqrt(R_true[0,0]), size=(1,1))  # generate with true noise
        x = A @ x + w
        y = H @ x + v
        # predict
        xhat_p = A @ xhat
        Pp = A @ P @ A.T + Q_filt
        if not np.all(np.isfinite(Pp)):
            return np.nan, np.nan, np.nan
        S = float((H @ Pp @ H.T)[0,0] + R_filt[0,0])  # filter believes R_filt
        if not np.isfinite(S) or S <= 0:
            S = 1e-12
        r_arr = y - H @ xhat_p
        r = float(r_arr.item())  # robust scalar extraction (avoids numpy deprecation)
        z2 = (r*r) / S
        # guard absurd values
        if not np.isfinite(z2) or z2 > cfg.get('nis_max', 1e6) or S > cfg.get('nis_max', 1e6):
            return np.nan, np.nan, np.nan
        # update using R_filt-consistent K
        xhat = xhat_p + K @ r_arr
        I2 = np.eye(n)
        P = (I2 - K @ H) @ Pp @ (I2 - K @ H).T + K @ R_filt @ K.T
        if not np.all(np.isfinite(P)):
            return np.nan, np.nan, np.nan
        if t >= burn:
            S_hist.append(S)
            z2_hist.append(z2)
    S_arr = np.array(S_hist)
    z2_arr = np.array(z2_hist)
    if len(z2_hist) == 0 or not np.all(np.isfinite(z2_hist)):
        return np.nan, np.nan, np.nan
    # windowed mean NIS (local calibration under non-normality)
    if cfg['nis_use_window'] and cfg['nis_window'] > 1 and len(z2_arr) > cfg['nis_window']:
        w = cfg['nis_window']
        nis_local = np.convolve(z2_arr, np.ones(w)/w, mode='valid')
        nis_mean = float(np.mean(nis_local))
    else:
        nis_mean = float(np.mean(z2_arr))
    nis_q = float(np.quantile(z2_arr, cfg['nis_quantile']))
    # event-style overconfidence relative to predictive S
    S_gamma = np.percentile(S_arr, gamma_q)
    over_rate = ( (z2_arr > (zstar**2)) & (S_arr < S_gamma) ).mean()
    return over_rate, nis_mean, nis_q

# === Hrisk ===
eps_values = make_eps_grid(CFG)
eps_ref = 0.3; Href = np.array([[1.0, eps_ref]])
# reference gain for normalization (use filter's R and Q)
K_ref, Pinf_ref = kalman_gain_ss(A, Href, Q_filt, R_filt)
Phiref = A - K_ref @ Href
Sigma_ref = Q_filt + K_ref @ R_filt @ K_ref.T
Pref, Mref = lyapunov_P(Phiref, Sigma_ref)
c1 = 1/(1 - spectral_radius(Phiref)); c2 = cond_number(Phiref)
c3 = integrated_sens(Mref); c4 = float((Href @ Pref @ Href.T)[0,0] / R_filt[0,0])

def Hrisk(Phi,H,P):
    rho=spectral_radius(Phi); kappa=cond_number(Phi)
    M=np.eye(4)-np.kron(Phi,Phi); intS=integrated_sens(M)
    IAI=float((H@P@H.T)[0,0] / R_filt[0,0])
    return (1/(1-rho)/c1)*(kappa/c2)*(intS/c3)*(IAI/c4)

Hrisks=[]; overs=[]

# --- adaptive sweep helpers ---
initial_zstar = 1.0
initial_gamma_q = 60

def run_sweep(zstar, gamma_q):
    Hrisks_local, over_local, nis_local, nisq_local = [], [], [], []
    # prepare fixed K if needed
    if CFG['k_policy'] == 'fixed_ref':
        K_fixed, _ = kalman_gain_ss(A, Href, Q_filt, R_filt)
    for eps in eps_values:
        H = np.array([[1.0, eps]])
        if CFG['k_policy'] == 'fixed_ref':
            K_use = K_fixed
            # evaluate Phi/P using fixed-K
            Phi = A - K_use @ H
            Sigma = Q_filt + K_use @ R_filt @ K_use.T
            P_ss, _ = lyapunov_P(Phi, Sigma)
        else:
            K_use, Pinf = kalman_gain_ss(A, H, Q_filt, R_filt)
            Phi = A - K_use @ H
            Sigma = Q_filt + K_use @ R_filt @ K_use.T
            P_ss, _ = lyapunov_P(Phi, Sigma)
        Hrisks_local.append(Hrisk(Phi, H, P_ss))
        over_rate, nis_mean, nis_q = simulate(
            A, H, Q_true, R_true, R_filt, Q_filt, K_use,
            zstar=zstar, gamma_q=gamma_q, cfg=CFG,
            w_seq=W_SEQ, v_seq=V_SEQ, T=T_SIM, burn=BURN
        )
        over_local.append(over_rate)
        nis_local.append(nis_mean)
        nisq_local.append(nis_q)
    return pd.DataFrame({'eps': eps_values,
                         'H_Risk': Hrisks_local,
                         'Overconf': over_local,
                         'NIS': nis_local,
                         'NIS_q': nisq_local}), zstar, gamma_q

def run_sweep_with_Hscale(zstar, gamma_q):
    rows = []
    scales = CFG['H_scale_list']
    for s in scales:
        # optional R scaling for SNR invariance
        R_true_eff, R_filt_eff = effective_Rs(R_true, R_filt, s, CFG['scale_R_with_H'])
        # prepare fixed-K if needed (note: gain is computed with the filter's belief and the scaled H)
        if CFG['k_policy'] == 'fixed_ref':
            K_fixed, _ = kalman_gain_ss(A, Href, Q_filt, R_filt_eff)
        for eps in eps_values:
            H_base = np.array([[1.0, eps]])
            Hs = s * H_base
            if CFG['k_policy'] == 'fixed_ref':
                K_use = K_fixed
                Phi = A - K_use @ Hs
                Sigma = Q_filt + K_use @ R_filt_eff @ K_use.T
                P_ss, _ = lyapunov_P(Phi, Sigma)
            else:
                K_use, Pinf = kalman_gain_ss(A, Hs, Q_filt, R_filt_eff)
                Phi = A - K_use @ Hs
                Sigma = Q_filt + K_use @ R_filt_eff @ K_use.T
                P_ss, _ = lyapunov_P(Phi, Sigma)
            Hrisk_val = Hrisk(Phi, Hs, P_ss)
            over_rate, nis_mean, nis_q = simulate(
                A, Hs, Q_true, R_true_eff, R_filt_eff, Q_filt, K_use,
                zstar=zstar, gamma_q=gamma_q, cfg=CFG,
                w_seq=W_SEQ, v_seq=V_SEQ, T=T_SIM, burn=BURN
            )
            rows.append({'scale': s, 'eps': eps, 'H_Risk': Hrisk_val, 'Overconf': over_rate, 'NIS': nis_mean, 'NIS_q': nis_q})
    df_scale = pd.DataFrame(rows)
    return df_scale

# 1st pass
df, z_used, gamma_used = run_sweep(initial_zstar, initial_gamma_q)

# If overconfidence is constant (all zeros), relax thresholds progressively
if df['Overconf'].std() == 0:
    for z_try, g_try in [(0.9, 70), (0.8, 80)]:
        df_try, z_used, gamma_used = run_sweep(z_try, g_try)
        if df_try['Overconf'].std() > 0:
            df = df_try
            break

# Validity summary
m_fin = np.isfinite(df['H_Risk'].values) & np.isfinite(df['NIS'].values)
print(f"Valid points (H_Risk & NIS finite): {int(m_fin.sum())}/{len(df)}; NaN dropped: {int((~m_fin).sum())}")

# === Stable correlation helpers ===
def corr_safe(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3 or np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return (np.nan, np.nan), (np.nan, np.nan)
    rp = pearsonr(x, y)
    rs = spearmanr(x, y)
    return rp, rs

# Density-robust: binned correlation on equal-size H_Risk bins
def binned_correlation(df, n_bins=10, col_x='H_Risk', col_y='NIS'):
    d = df[[col_x, col_y]].copy()
    d = d[np.isfinite(d[col_x].values) & np.isfinite(d[col_y].values)]
    if len(d) < 3:
        return np.nan
    bins = np.linspace(d[col_x].min(), d[col_x].max(), n_bins + 1)
    # digitize returns bin indices in 1..n_bins+1; keep within 1..n_bins
    idx = np.digitize(d[col_x].values, bins, right=False)
    idx = np.clip(idx, 1, n_bins)
    gb = pd.DataFrame({'x': d[col_x].values, 'y': d[col_y].values, 'bin': idx})
    grouped = gb.groupby('bin')[['x','y']].mean()
    if len(grouped) < 3 or np.allclose(grouped['x'].std(), 0) or np.allclose(grouped['y'].std(), 0):
        return np.nan
    r, _ = pearsonr(grouped['x'].values, grouped['y'].values)
    return float(r)

# Compute correlations (primary: NIS; secondary: Overconfidence if non-constant)
if df['Overconf'].std() == 0:
    r_p_over = float('nan'); p_p_over = float('nan')
    r_s_over = float('nan'); p_s_over = float('nan')
else:
    (r_p_over, p_p_over), (r_s_over, p_s_over) = corr_safe(df['H_Risk'], df['Overconf'])

(r_p_nis, p_p_nis), (r_s_nis, p_s_nis) = corr_safe(df['H_Risk'], df['NIS'])

print("Overconfidence: Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)" % (r_p_over, p_p_over, r_s_over, p_s_over))
print("NIS:           Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)" % (r_p_nis, p_p_nis, r_s_nis, p_s_nis))

r_bin_nis = binned_correlation(df, n_bins=10, col_x='H_Risk', col_y='NIS')
print(f"NIS (binned 10):   Pearson r={r_bin_nis:.3f}")

(r_p_nisq, p_p_nisq), (r_s_nisq, p_s_nisq) = corr_safe(df['H_Risk'], df['NIS_q'])
print("NIS_q:        Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)" % (r_p_nisq, p_p_nisq, r_s_nisq, p_s_nisq))

# --- Bootstrap CIs for correlation and slope (publication-ready) ---
def bootstrap_ci_xy(x, y, stat_fn, B=1000, seed=123):
    rng_bs = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < 3 or np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return (np.nan, np.nan)
    vals = []
    for _ in range(B):
        idx = rng_bs.integers(0, n, n)
        vals.append(stat_fn(x[idx], y[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

# Stats on mean NIS
pearson_stat = lambda x,y: pearsonr(x,y)[0]
ts_slope_stat = lambda x,y: theilslopes(y, x)[0]
nis_ci_pear = bootstrap_ci_xy(df['H_Risk'].values, df['NIS'].values, pearson_stat)
nis_ci_ts   = bootstrap_ci_xy(df['H_Risk'].values, df['NIS'].values, ts_slope_stat)
print(f"NIS bootstrap 95% CI: Pearson r in {nis_ci_pear}, TS slope in {nis_ci_ts}")

# Stats on tail NIS_q
nisq_ci_pear = bootstrap_ci_xy(df['H_Risk'].values, df['NIS_q'].values, pearson_stat)
nisq_ci_ts   = bootstrap_ci_xy(df['H_Risk'].values, df['NIS_q'].values, ts_slope_stat)
print(f"NIS_q bootstrap 95% CI: Pearson r in {nisq_ci_pear}, TS slope in {nisq_ci_ts}")

# === Influence diagnostics: outlier / leverage guarding ===
# Identify potential leverage point: the minimum H_Risk (often the reference epsilon)
idx_min = int(df['H_Risk'].idxmin())
mask_wo_min = np.ones(len(df), dtype=bool); mask_wo_min[idx_min] = False

# Jackknife (leave-one-out) Pearson on NIS
jack_r = []
for i in range(len(df)):
    m = np.ones(len(df), dtype=bool); m[i] = False
    x = df.loc[m, 'H_Risk'].values; y = df.loc[m, 'NIS'].values
    mm = np.isfinite(x) & np.isfinite(y)
    x = x[mm]; y = y[mm]
    if x.size < 3 or np.allclose(x.std(),0) or np.allclose(y.std(),0):
        jack_r.append(np.nan)
    else:
        r,_ = pearsonr(x, y)
        jack_r.append(r)
jack_r = np.array(jack_r)

# Correlations without the minimum H_Risk point
if df.loc[mask_wo_min, 'NIS'].std() > 0:
    r_p_nis_wo, p_p_nis_wo = pearsonr(df.loc[mask_wo_min, 'H_Risk'], df.loc[mask_wo_min, 'NIS'])
    r_s_nis_wo, p_s_nis_wo = spearmanr(df.loc[mask_wo_min, 'H_Risk'], df.loc[mask_wo_min, 'NIS'])
else:
    r_p_nis_wo = r_s_nis_wo = np.nan
    p_p_nis_wo = p_s_nis_wo = np.nan

# Theil--Sen slope with finite mask
_mx = np.isfinite(df['H_Risk'].values) & np.isfinite(df['NIS'].values)
if _mx.sum() >= 3:
    ts_slope, ts_inter, _, _ = theilslopes(df['NIS'].values[_mx], df['H_Risk'].values[_mx])
else:
    ts_slope, ts_inter = np.nan, np.nan
_mx_wo = np.isfinite(df.loc[mask_wo_min, 'H_Risk'].values) & np.isfinite(df.loc[mask_wo_min, 'NIS'].values)
if _mx_wo.sum() >= 3:
    ts_slope_wo, ts_inter_wo, _, _ = theilslopes(df.loc[mask_wo_min, 'NIS'].values[_mx_wo], df.loc[mask_wo_min, 'H_Risk'].values[_mx_wo])
else:
    ts_slope_wo, ts_inter_wo = np.nan, np.nan

print("Jackknife Pearson on NIS (min/median/max):", np.nanmin(jack_r), np.nanmedian(jack_r), np.nanmax(jack_r))
print("NIS w/o min(H_Risk): Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)" % (r_p_nis_wo, p_p_nis_wo, r_s_nis_wo, p_s_nis_wo))
print("Theil--Sen slope (full / w/o min):", ts_slope, ts_slope_wo)

# === Plot ===
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(df.eps,df.H_Risk); plt.xscale("log")
set_labels(
    plt.gca(),
    'Observability coupling  $\\varepsilon$',
    'Instability index  $H_{\\mathrm{Risk}}$ (unitless)',
    None
)

plt.subplot(1,2,2)
plt.scatter(df.H_Risk, df.NIS)
# Compute robust Theil--Sen line (full)
from scipy.stats import linregress
xline=np.linspace(df['H_Risk'].min(), df['H_Risk'].max(), 100)
if not np.isnan(ts_slope) and not np.isnan(ts_inter):
    plt.plot(xline, ts_inter + ts_slope*xline, linestyle='--', color='C1', label='Theil--Sen (robust)')

# Compute OLS for reporting only (do not plot)
ols_res = linregress(df['H_Risk'], df['NIS'])
OLS_slope_NIS, OLS_intercept_NIS = ols_res.slope, ols_res.intercept
TS_slope_NIS, TS_intercept_NIS = ts_slope, ts_inter
if TS_slope_NIS is not None and np.isfinite(TS_slope_NIS) and not np.isclose(TS_slope_NIS, 0.0):
    OLS_vs_TS_slope_diff_pct_NIS = 100.0 * abs(OLS_slope_NIS - TS_slope_NIS) / abs(TS_slope_NIS)
else:
    OLS_vs_TS_slope_diff_pct_NIS = np.nan
print(f"[NIS] TS slope={TS_slope_NIS:.6g}, TS intercept={TS_intercept_NIS:.6g}; OLS slope={OLS_slope_NIS:.6g}, OLS intercept={OLS_intercept_NIS:.6g}; rel diff={OLS_vs_TS_slope_diff_pct_NIS:.3f}%")

# Legend (Theil--Sen only)
if any(l.get_label() == 'Theil--Sen (robust)' for l in plt.gca().lines):
    plt.legend()

set_labels(
    plt.gca(),
    'Instability index  $H_{\\mathrm{Risk}}$ (unitless)',
    'Calibration  NIS $= \\mathbb{E}[Z^2]$',
    None
)
plt.tight_layout()
plt.savefig("LTI_dual_plot_autotuned.png", dpi=CFG.get('plot_dpi', 300))

# Extra figure: H_Risk vs NIS upper-quantile
plt.figure(figsize=(5,4))
plt.scatter(df.H_Risk, df.NIS_q)
# Theil--Sen line for NIS_q
_mq = np.isfinite(df['H_Risk'].values) & np.isfinite(df['NIS_q'].values)
if _mq.sum() >= 3:
    ts_slope_q, ts_inter_q, _, _ = theilslopes(df['NIS_q'].values[_mq], df['H_Risk'].values[_mq])
else:
    ts_slope_q, ts_inter_q = np.nan, np.nan
xline=np.linspace(df['H_Risk'].min(), df['H_Risk'].max(), 100)
if not np.isnan(ts_slope_q) and not np.isnan(ts_inter_q):
    plt.plot(xline, ts_inter_q + ts_slope_q*xline, linestyle='--', color='C1', label='Theil--Sen (robust)')

# OLS for reporting only (do not plot)
ols_res_q = linregress(df['H_Risk'], df['NIS_q'])
OLS_slope_NIS_q, OLS_intercept_NIS_q = ols_res_q.slope, ols_res_q.intercept
TS_slope_NIS_q, TS_intercept_NIS_q = ts_slope_q, ts_inter_q
if TS_slope_NIS_q is not None and np.isfinite(TS_slope_NIS_q) and not np.isclose(TS_slope_NIS_q, 0.0):
    OLS_vs_TS_slope_diff_pct_NIS_q = 100.0 * abs(OLS_slope_NIS_q - TS_slope_NIS_q) / abs(TS_slope_NIS_q)
else:
    OLS_vs_TS_slope_diff_pct_NIS_q = np.nan
print(f"[NIS_q] TS slope={TS_slope_NIS_q:.6g}, TS intercept={TS_intercept_NIS_q:.6g}; OLS slope={OLS_slope_NIS_q:.6g}, OLS intercept={OLS_intercept_NIS_q:.6g}; rel diff={OLS_vs_TS_slope_diff_pct_NIS_q:.3f}%")

if any(l.get_label() == 'Theil--Sen (robust)' for l in plt.gca().lines):
    plt.legend()
set_labels(
    plt.gca(),
    'Instability index  $H_{\\mathrm{Risk}}$ (unitless)',
    f'Tail calibration  NIS quantile  q={CFG["nis_quantile"]}',
    None
)
plt.tight_layout(); plt.savefig('LTI_Hrisk_vs_NISq.png', dpi=CFG.get('plot_dpi', 300))

# === H-scale sweeps: compare SNR-invariant vs SNR-changing ===
# First run with current CFG['scale_R_with_H'] setting
df_scale = run_sweep_with_Hscale(z_used, gamma_used)
# Correlations per scale
corrs = []
for s in CFG['H_scale_list']:
    d = df_scale[df_scale['scale']==s]
    _dx = d['H_Risk'].values
    _dy = d['NIS'].values
    _m = np.isfinite(_dx) & np.isfinite(_dy)
    if _m.sum() >= 3 and not np.allclose(np.std(_dx[_m]), 0) and not np.allclose(np.std(_dy[_m]), 0):
        rp, pp = pearsonr(_dx[_m], _dy[_m])
        rs, ps = spearmanr(_dx[_m], _dy[_m])
    else:
        rp, pp = (np.nan, np.nan)
        rs, ps = (np.nan, np.nan)
    print(f"[finite-check] used {int(_m.sum())}/{len(_m)} points for scale s={s}")
    corrs.append((s, rp, pp, rs, ps))
print("H-scale sweep (scale_R_with_H=%s):" % CFG['scale_R_with_H'])
for s, rp, pp, rs, ps in corrs:
    print("  s=%.2f: NIS vs H_Risk Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)" % (s, rp, pp, rs, ps))

# Plot: color-coded by scale
plt.figure(figsize=(6,5))
for s in CFG['H_scale_list']:
    d = df_scale[df_scale['scale']==s]
    plt.scatter(d['H_Risk'], d['NIS'], label=f's={s}')
set_labels(
    plt.gca(),
    'Instability index  $H_{\\mathrm{Risk}}$ (unitless)',
    'Calibration  NIS $= \\mathbb{E}[Z^2]$',
    None
)
# Add overall Theil--Sen line for the combined H-scale sweep (display only TS)
_m_scale = np.isfinite(df_scale['H_Risk'].values) & np.isfinite(df_scale['NIS'].values)
if _m_scale.sum() >= 3:
    ts_slope_scale, ts_inter_scale, _, _ = theilslopes(df_scale['NIS'].values[_m_scale], df_scale['H_Risk'].values[_m_scale])
    xs = np.linspace(df_scale['H_Risk'].min(), df_scale['H_Risk'].max(), 100)
    plt.plot(xs, ts_inter_scale + ts_slope_scale*xs, linestyle='--', color='k', label='Theil--Sen (robust)')
    plt.legend()
plt.legend(); plt.tight_layout(); plt.savefig('LTI_Hscale_vs_NIS.png', dpi=CFG.get('plot_dpi', 300))

# Also run the complementary mode once (toggle scale_R_with_H) to illustrate invariance vs change
SCALE_R_INIT = CFG['scale_R_with_H']
CFG['scale_R_with_H'] = not CFG['scale_R_with_H']
df_scale2 = run_sweep_with_Hscale(z_used, gamma_used)
corrs2 = []
for s in CFG['H_scale_list']:
    d = df_scale2[df_scale2['scale']==s]
    _dx = d['H_Risk'].values
    _dy = d['NIS'].values
    _m = np.isfinite(_dx) & np.isfinite(_dy)
    if _m.sum() >= 3 and not np.allclose(np.std(_dx[_m]), 0) and not np.allclose(np.std(_dy[_m]), 0):
        rp, pp = pearsonr(_dx[_m], _dy[_m])
        rs, ps = spearmanr(_dx[_m], _dy[_m])
    else:
        rp, pp = (np.nan, np.nan)
        rs, ps = (np.nan, np.nan)
    print(f"[finite-check] used {int(_m.sum())}/{len(_m)} points for scale s={s}")
    corrs2.append((s, rp, pp, rs, ps))
print("H-scale sweep (scale_R_with_H=%s):" % CFG['scale_R_with_H'])
for s, rp, pp, rs, ps in corrs2:
    print("  s=%.2f: NIS vs H_Risk Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)" % (s, rp, pp, rs, ps))

plt.figure(figsize=(6,5))
for s in CFG['H_scale_list']:
    d = df_scale2[df_scale2['scale']==s]
    plt.scatter(d['H_Risk'], d['NIS'], label=f's={s}')
set_labels(
    plt.gca(),
    'Instability index  $H_{\\mathrm{Risk}}$ (unitless)',
    'Calibration  NIS $= \\mathbb{E}[Z^2]$',
    None
)
# Add overall Theil--Sen line for toggled sweep
_m_scale2 = np.isfinite(df_scale2['H_Risk'].values) & np.isfinite(df_scale2['NIS'].values)
if _m_scale2.sum() >= 3:
    ts_slope_scale2, ts_inter_scale2, _, _ = theilslopes(df_scale2['NIS'].values[_m_scale2], df_scale2['H_Risk'].values[_m_scale2])
    xs2 = np.linspace(df_scale2['H_Risk'].min(), df_scale2['H_Risk'].max(), 100)
    plt.plot(xs2, ts_inter_scale2 + ts_slope_scale2*xs2, linestyle='--', color='k', label='Theil--Sen (robust)')
    plt.legend()
plt.legend(); plt.tight_layout(); plt.savefig('LTI_Hscale_vs_NIS_toggle.png', dpi=CFG.get('plot_dpi', 300))

# === Supplemental experiments for paper figures ===

def run_sweep_generic(A_local, k_policy_local, tag_suffix, zstar=initial_zstar, gamma_q=initial_gamma_q):
    """Run a full sweep with a local A and k_policy, produce NIS and NIS_q figures.
       Saves: f'supp_{tag_suffix}_NIS.png', f'supp_{tag_suffix}_NISq.png'.
    """
    # Build reference constants for Hrisk (local to A_local)
    eps_vals = eps_values
    Href_loc = np.array([[1.0, 0.3]])
    K_ref_loc, _ = kalman_gain_ss(A_local, Href_loc, Q_filt, R_filt)
    Phiref_loc = A_local - K_ref_loc @ Href_loc
    Sigma_ref_loc = Q_filt + K_ref_loc @ R_filt @ K_ref_loc.T
    Pref_loc, Mref_loc = lyapunov_P(Phiref_loc, Sigma_ref_loc)
    c1_loc = 1/(1 - spectral_radius(Phiref_loc))
    c2_loc = cond_number(Phiref_loc)
    c3_loc = integrated_sens(Mref_loc)
    c4_loc = float((Href_loc @ Pref_loc @ Href_loc.T)[0,0] / R_filt[0,0])

    def Hrisk_loc(Phi,H,P):
        rho=spectral_radius(Phi); kappa=cond_number(Phi)
        M=np.eye(4)-np.kron(Phi,Phi); intS=integrated_sens(M)
        IAI=float((H@P@H.T)[0,0] / R_filt[0,0])
        return (1/(1-rho)/c1_loc)*(kappa/c2_loc)*(intS/c3_loc)*(IAI/c4_loc)

    # Prepare K if fixed
    if k_policy_local == 'fixed_ref':
        K_fixed_loc, _ = kalman_gain_ss(A_local, Href_loc, Q_filt, R_filt)

    rows = []
    for eps in eps_vals:
        Hloc = np.array([[1.0, eps]])
        if k_policy_local == 'fixed_ref':
            K_use = K_fixed_loc
        else:
            K_use, _ = kalman_gain_ss(A_local, Hloc, Q_filt, R_filt)
        Phi = A_local - K_use @ Hloc
        Sigma = Q_filt + K_use @ R_filt @ K_use.T
        P_ss, _ = lyapunov_P(Phi, Sigma)
        hr = Hrisk_loc(Phi, Hloc, P_ss)
        over_rate, nis_mean, nis_q = simulate(
            A_local, Hloc, Q_true, R_true, R_filt, Q_filt, K_use,
            zstar=zstar, gamma_q=gamma_q, cfg=CFG,
            w_seq=W_SEQ, v_seq=V_SEQ, T=T_SIM, burn=BURN
        )
        rows.append({'eps': eps, 'H_Risk': hr, 'Overconf': over_rate, 'NIS': nis_mean, 'NIS_q': nis_q})

    dfg = pd.DataFrame(rows)
    m_fin = np.isfinite(dfg['H_Risk'].values) & np.isfinite(dfg['NIS'].values)
    print(f"[supp {tag_suffix}] valid: {int(m_fin.sum())}/{len(dfg)} (finite H_Risk & NIS)")
    (rp, pp), (rs, ps) = corr_safe(dfg['H_Risk'], dfg['NIS'])
    print(f"[supp {tag_suffix}] NIS: Pearson r={rp:.3f} (p={pp:.4f}), Spearman r={rs:.3f} (p={ps:.4f})")

    # Plot NIS (Theil--Sen only)
    plt.figure(figsize=(5.6,4.4))
    plt.scatter(dfg['H_Risk'], dfg['NIS'])
    _m = np.isfinite(dfg['H_Risk'].values) & np.isfinite(dfg['NIS'].values)
    if _m.sum() >= 3:
        ts_sl, ts_it, _, _ = theilslopes(dfg['NIS'].values[_m], dfg['H_Risk'].values[_m])
        xs = np.linspace(dfg['H_Risk'].min(), dfg['H_Risk'].max(), 100)
        plt.plot(xs, ts_it + ts_sl*xs, linestyle='--', color='C1', label='Theil--Sen (robust)')
        plt.legend()
    set_labels(
        plt.gca(),
        'Instability index  $H_{\\mathrm{Risk}}$ (unitless)',
        'Calibration  NIS $= \\mathbb{E}[Z^2]$',
        None
    )
    plt.tight_layout(); plt.savefig(f'supp_{tag_suffix}_NIS.png', dpi=CFG.get('plot_dpi', 300))

    # Plot NIS_q (Theil--Sen only)
    (rpq, ppq), _ = corr_safe(dfg['H_Risk'], dfg['NIS_q'])
    plt.figure(figsize=(5.6,4.4))
    plt.scatter(dfg['H_Risk'], dfg['NIS_q'])
    _mq = np.isfinite(dfg['H_Risk'].values) & np.isfinite(dfg['NIS_q'].values)
    if _mq.sum() >= 3:
        ts_sl2, ts_it2, _, _ = theilslopes(dfg['NIS_q'].values[_mq], dfg['H_Risk'].values[_mq])
        xs2 = np.linspace(dfg['H_Risk'].min(), dfg['H_Risk'].max(), 100)
        plt.plot(xs2, ts_it2 + ts_sl2*xs2, linestyle='--', color='C1', label='Theil--Sen (robust)')
        plt.legend()
    set_labels(
        plt.gca(),
        'Instability index  $H_{\\mathrm{Risk}}$ (unitless)',
        f'Tail calibration  NIS quantile  q={CFG["nis_quantile"]}',
        None
    )
    plt.tight_layout(); plt.savefig(f'supp_{tag_suffix}_NISq.png', dpi=CFG.get('plot_dpi', 300))
    return dfg

# (1) A[0,1] = 0.75, keep current k_policy
A_075 = A.copy(); A_075[0,1] = 0.75
_ = run_sweep_generic(A_075, CFG['k_policy'], tag_suffix='A12_0p75_'+CFG['k_policy'])

# (2) k_policy back to 'dare_per_eps' as a control (with current A)
_ = run_sweep_generic(A, 'dare_per_eps', tag_suffix='kpolicy_dare')


 # === Supplemental sweeps: misspecification in Q_filt and R_filt, and A mismatch ===

def run_sweep_misspec_Q(gammas=(0.08,0.10,0.12,0.15,0.20,0.30,0.50), tag='Qfilt_sweep'):
    rows=[]
    # Reference K at a nominal eps (0.3) with rescaled Qf
    for g in gammas:
        Qf = Q_filt * (g/CFG['Q_filt_scale'])  # rescale relative to current baseline
        Kref,_ = kalman_gain_ss(A, np.array([[1.0, 0.3]]), Qf, R_filt)
        for eps in eps_values:
            Hx = np.array([[1.0, eps]])
            Kuse = Kref if CFG['k_policy']=='fixed_ref' else kalman_gain_ss(A,Hx,Qf,R_filt)[0]
            Phi = A - Kuse @ Hx
            Sigma = Q_filt + Kuse @ R_filt @ Kuse.T
            Pss,_ = lyapunov_P(Phi, Sigma)
            hr = Hrisk(Phi, Hx, Pss)
            _, nis_mean, nis_q = simulate(
                A, Hx, Q_true, R_true, R_filt, Qf, Kuse,
                zstar=initial_zstar, gamma_q=initial_gamma_q, cfg=CFG,
                w_seq=W_SEQ, v_seq=V_SEQ, T=T_SIM, burn=BURN
            )
            rows.append({'scaleQ': g, 'eps': eps, 'H_Risk': hr, 'NIS': nis_mean, 'NIS_q': nis_q})
    df = pd.DataFrame(rows)
    # plots
    plt.figure(figsize=(6,4.5))
    for g in sorted(df['scaleQ'].unique()):
        d = df[df['scaleQ']==g]
        plt.scatter(d['H_Risk'], d['NIS'], s=18, label=f"Qfilt={g}")
    plt.legend(); set_labels(plt.gca(), 'Instability index  $H_{\\mathrm{Risk}}$ (unitless)', 'Calibration  NIS $= \\mathbb{E}[Z^2]$', f'supp: {tag} (NIS)')
    plt.tight_layout(); plt.savefig(f'supp_{tag}_NIS.png', dpi=CFG.get('plot_dpi',300))

    plt.figure(figsize=(6,4.5))
    for g in sorted(df['scaleQ'].unique()):
        d = df[df['scaleQ']==g]
        plt.scatter(d['H_Risk'], d['NIS_q'], s=18, label=f"Qfilt={g}")
    plt.legend(); set_labels(plt.gca(), 'Instability index  $H_{\\mathrm{Risk}}$ (unitless)', f'Tail calibration  NIS quantile  q={CFG["nis_quantile"]}', f'supp: {tag} (NIS_q)')
    plt.tight_layout(); plt.savefig(f'supp_{tag}_NISq.png', dpi=CFG.get('plot_dpi',300))
    m_fin = np.isfinite(df['H_Risk'].values) & np.isfinite(df['NIS'].values)
    print(f"[supp {tag}] valid: {int(m_fin.sum())}/{len(df)} (finite H_Risk & NIS)")
    return df


def run_sweep_misspec_R(gammas=(0.20,0.25,0.30,0.40,0.50), tag='Rfilt_sweep'):
    rows=[]
    for g in gammas:
        Rf = R_filt * (g/CFG['R_filt_scale'])  # rescale relative baseline
        Kref,_ = kalman_gain_ss(A, np.array([[1.0, 0.3]]), Q_filt, Rf)
        for eps in eps_values:
            Hx = np.array([[1.0, eps]])
            Kuse = Kref if CFG['k_policy']=='fixed_ref' else kalman_gain_ss(A,Hx,Q_filt,Rf)[0]
            Phi = A - Kuse @ Hx
            Sigma = Q_filt + Kuse @ Rf @ Kuse.T
            Pss,_ = lyapunov_P(Phi, Sigma)
            hr = Hrisk(Phi, Hx, Pss)
            _, nis_mean, nis_q = simulate(
                A, Hx, Q_true, R_true, Rf, Q_filt, Kuse,
                zstar=initial_zstar, gamma_q=initial_gamma_q, cfg=CFG,
                w_seq=W_SEQ, v_seq=V_SEQ, T=T_SIM, burn=BURN
            )
            rows.append({'scaleR': g, 'eps': eps, 'H_Risk': hr, 'NIS': nis_mean, 'NIS_q': nis_q})
    df = pd.DataFrame(rows)
    # plots
    plt.figure(figsize=(6,4.5))
    for g in sorted(df['scaleR'].unique()):
        d = df[df['scaleR']==g]
        plt.scatter(d['H_Risk'], d['NIS'], s=18, label=f"Rfilt={g}")
    plt.legend(); set_labels(plt.gca(), 'Instability index  $H_{\\mathrm{Risk}}$ (unitless)', 'Calibration  NIS $= \\mathbb{E}[Z^2]$', f'supp: {tag} (NIS)')
    plt.tight_layout(); plt.savefig(f'supp_{tag}_NIS.png', dpi=CFG.get('plot_dpi',300))

    plt.figure(figsize=(6,4.5))
    for g in sorted(df['scaleR'].unique()):
        d = df[df['scaleR']==g]
        plt.scatter(d['H_Risk'], d['NIS_q'], s=18, label=f"Rfilt={g}")
    plt.legend(); set_labels(plt.gca(), 'Instability index  $H_{\\mathrm{Risk}}$ (unitless)', f'Tail calibration  NIS quantile  q={CFG["nis_quantile"]}', f'supp: {tag} (NIS_q)')
    plt.tight_layout(); plt.savefig(f'supp_{tag}_NISq.png', dpi=CFG.get('plot_dpi',300))
    m_fin = np.isfinite(df['H_Risk'].values) & np.isfinite(df['NIS'].values)
    print(f"[supp {tag}] valid: {int(m_fin.sum())}/{len(df)} (finite H_Risk & NIS)")
    return df


def run_sweep_A_mismatch(delta_norms=(0.0, 0.01, 0.02), tag='A_mismatch'):
    rows=[]
    for dn in delta_norms:
        # construct a small bias in A (upper-triangular perturbation scaled to Fro norm dn)
        Delta = np.array([[0.0, 0.1],[0.0, 0.0]])
        if dn>0:
            fn = np.linalg.norm(Delta, 'fro')
            if fn>0: Delta = Delta * (dn/fn)
        A_true_loc = A + Delta
        # recompute K_ref at nominal (fixed_ref uses nominal A)
        Kref,_ = kalman_gain_ss(A, np.array([[1.0, 0.3]]), Q_filt, R_filt)
        for eps in eps_values:
            Hx = np.array([[1.0, eps]])
            Kuse = Kref if CFG['k_policy']=='fixed_ref' else kalman_gain_ss(A,Hx,Q_filt,R_filt)[0]
            Phi = A_true_loc - Kuse @ Hx   # true dynamics in loop
            Sigma = Q_filt + Kuse @ R_filt @ Kuse.T
            Pss,_ = lyapunov_P(Phi, Sigma)
            hr = Hrisk(Phi, Hx, Pss)
            _, nis_mean, nis_q = simulate(
                A_true_loc, Hx, Q_true, R_true, R_filt, Q_filt, Kuse,
                zstar=initial_zstar, gamma_q=initial_gamma_q, cfg=CFG,
                w_seq=W_SEQ, v_seq=V_SEQ, T=T_SIM, burn=BURN
            )
            rows.append({'deltaA': dn, 'eps': eps, 'H_Risk': hr, 'NIS': nis_mean, 'NIS_q': nis_q})
    df = pd.DataFrame(rows)
    # plots
    plt.figure(figsize=(6,4.5))
    for dn in sorted(df['deltaA'].unique()):
        d = df[df['deltaA']==dn]
        plt.scatter(d['H_Risk'], d['NIS'], s=18, label=f"||ΔA||_F={dn}")
    plt.legend(); set_labels(plt.gca(), 'Instability index  $H_{\\mathrm{Risk}}$ (unitless)', 'Calibration  NIS $= \\mathbb{E}[Z^2]$ ', f'supp: {tag} (NIS)')
    plt.tight_layout(); plt.savefig(f'supp_{tag}_NIS.png', dpi=CFG.get('plot_dpi',300))

    plt.figure(figsize=(6,4.5))
    for dn in sorted(df['deltaA'].unique()):
        d = df[df['deltaA']==dn]
        plt.scatter(d['H_Risk'], d['NIS_q'], s=18, label=f"||ΔA||_F={dn}")
    plt.legend(); set_labels(plt.gca(), 'Instability index  $H_{\\mathrm{Risk}}$ (unitless)', f'Tail calibration  NIS quantile  q={CFG["nis_quantile"]}', f'supp: {tag} (NIS_q)')
    plt.tight_layout(); plt.savefig(f'supp_{tag}_NISq.png', dpi=CFG.get('plot_dpi',300))
    m_fin = np.isfinite(df['H_Risk'].values) & np.isfinite(df['NIS'].values)
    print(f"[supp {tag}] valid: {int(m_fin.sum())}/{len(df)} (finite H_Risk & NIS)")
    return df

# Conditionally run supplemental variants
if CFG.get('run_supplemental_variants', True):
    _ = run_sweep_misspec_Q()
    _ = run_sweep_misspec_R()
    _ = run_sweep_A_mismatch()

# Persist a machine-readable summary.
sumrow = {
    'pearson_NIS': r_p_nis, 'spearman_NIS': r_s_nis,
    'pearson_NIS_CI_lo': nis_ci_pear[0], 'pearson_NIS_CI_hi': nis_ci_pear[1],
    'TS_slope_NIS': ts_slope, 'TS_intercept_NIS': ts_inter,
    'TS_slope_NIS_CI_lo': nis_ci_ts[0], 'TS_slope_NIS_CI_hi': nis_ci_ts[1],
    'OLS_slope_NIS': OLS_slope_NIS, 'OLS_intercept_NIS': OLS_intercept_NIS,
    'OLS_vs_TS_slope_diff_pct_NIS': OLS_vs_TS_slope_diff_pct_NIS,
    'pearson_NIS_binned_10': r_bin_nis,
    'pearson_NISq': r_p_nisq, 'spearman_NISq': r_s_nisq,
    'pearson_NISq_CI_lo': nisq_ci_pear[0], 'pearson_NISq_CI_hi': nisq_ci_pear[1],
    'TS_slope_NIS_q': TS_slope_NIS_q, 'TS_intercept_NIS_q': TS_intercept_NIS_q,
    'OLS_slope_NIS_q': OLS_slope_NIS_q, 'OLS_intercept_NIS_q': OLS_intercept_NIS_q,
    'OLS_vs_TS_slope_diff_pct_NIS_q': OLS_vs_TS_slope_diff_pct_NIS_q,
    # Experiment manifest
    'seed': CFG['seed'],
    'rho_cutoff': CFG['rho_cutoff'],
    'nis_max': CFG['nis_max'],
    'nis_quantile': CFG['nis_quantile'],
    'nis_window': CFG['nis_window'],
    'nis_use_window': CFG['nis_use_window'],
    'Q_filt_scale': CFG['Q_filt_scale'],
    'R_filt_scale': CFG['R_filt_scale'],
    'k_policy': CFG['k_policy'],
    'scale_R_with_H_initial': SCALE_R_INIT,
    'A12': float(A[0,1]),
    'T_SIM': T_SIM,
    'BURN': BURN,
}
pd.DataFrame([sumrow]).to_csv('LTI_summary_stats.csv', index=False)

# Save artifacts next to this script
df['valid_mask'] = np.isfinite(df['H_Risk'].values) & np.isfinite(df['NIS'].values)
df.to_csv("LTI_results_autotuned.csv", index=False)
df_scale.to_csv('LTI_scale_mode1.csv', index=False)
df_scale2.to_csv('LTI_scale_mode2.csv', index=False)
