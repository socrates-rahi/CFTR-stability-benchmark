#!/usr/bin/env python3
"""
Consensus / calibration model for CFTR NBD stability predictors.

Predicts experimental dTm (deg C) from predictor outputs (ddG-like scores) and is
evaluated with LEAVE-ONE-POSITION-OUT cross-validation (all substitutions at the same
residue of the same domain are held out together, so e.g. A1402H on four NBD2
constructs can never leak into itself).

Usage:
  python3 consensus_model.py bahia_reference_59.csv
  python3 consensus_model.py benchmark.csv --features foldx,rosetta_hrp,ddmut_fwd --flip ddmut_fwd

--flip: columns whose sign convention is "positive = stabilising" (e.g. mCSM-family tools,
        possibly DDMut -- CHECK the server output page). They are negated so that every
        feature means "positive = destabilising", like FoldX/Rosetta ddG.
Input columns needed: domain, mutation (e.g. S492F), exp_dTm, plus the feature columns.
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument('csv')
p.add_argument('--features', default='foldx,rosetta_hrp,rosetta_lrp')
p.add_argument('--flip', default='')
p.add_argument('--out', default='results')
p.add_argument('--boot', type=int, default=2000)
p.add_argument('--transform', choices=['slog','none'], default='slog')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

df = pd.read_csv(a.csv)
feats = [f for f in a.features.split(',') if f]
flip = [f for f in a.flip.split(',') if f]
for f in feats:
    if f not in df: sys.exit(f'missing column {f}')
df = df.dropna(subset=feats + ['exp_dTm']).reset_index(drop=True)
for f in flip: df[f] = -df[f]

df['pos'] = df['mutation'].str.extract(r'(\d+)')[0]
df['group'] = df['domain'] + '_' + df['pos']
y = df['exp_dTm'].values
G = df['group'].values
isN2 = (df['domain'] == 'NBD2').astype(float).values

def slog(x):  # signed log: tames the 100+ kcal/mol Rosetta outliers, keeps sign
    return np.sign(x) * np.log1p(np.abs(x))
X = np.column_stack([(slog(df[f].values) if a.transform=='slog' else df[f].values) for f in feats])

def fit_ridge(Xtr, ytr, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Z = (Xtr - mu) / sd
    ym = ytr.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (ytr - ym))
    return lambda Xte: ((Xte - mu) / sd) @ w + ym

def with_domain(Xm, d): return np.column_stack([Xm, d])

def make_model(kind, cols):
    """returns a function (Xtr,dtr,ytr,gtr) -> predictor(Xte,dte)"""
    def build(Xtr, dtr, ytr, gtr):
        if kind == 'single':
            Xs = with_domain(Xtr[:, cols], dtr)
            m = fit_ridge(Xs, ytr, 1e-6)
            return lambda Xte, dte: m(with_domain(Xte[:, cols], dte))
        if kind == 'zmean':   # equal-weight consensus: mean of standardised features, then calibrate
            mu, sd = Xtr[:, cols].mean(0), Xtr[:, cols].std(0); sd[sd == 0] = 1
            s = lambda Xm: ((Xm[:, cols] - mu) / sd).mean(1)
            m = fit_ridge(with_domain(s(Xtr)[:, None], dtr), ytr, 1e-6)
            return lambda Xte, dte: m(with_domain(s(Xte)[:, None], dte))
        if kind == 'ridge':   # stacked regression; penalty chosen by inner leave-one-group-out
            best, besta = 1e18, 1.0
            for al in (0.3, 1, 3, 10, 30, 100):
                err = 0
                for g in np.unique(gtr):
                    t = gtr != g
                    m = fit_ridge(with_domain(Xtr[t][:, cols], dtr[t]), ytr[t], al)
                    err += ((m(with_domain(Xtr[~t][:, cols], dtr[~t])) - ytr[~t]) ** 2).sum()
                if err < best: best, besta = err, al
            m = fit_ridge(with_domain(Xtr[:, cols], dtr), ytr, besta)
            return lambda Xte, dte: m(with_domain(Xte[:, cols], dte))
    return build

models = {}
for i, f in enumerate(feats):
    models[f'{f} (calibrated)'] = make_model('single', [i])
if len(feats) > 1:
    models['z-score mean consensus'] = make_model('zmean', list(range(len(feats))))
    models['ridge stack (all features)'] = make_model('ridge', list(range(len(feats))))

# ---- leave-one-position-out predictions -----------------------------------------------
oof = {}
groups = np.unique(G)
for name, build in models.items():
    pred = np.zeros(len(y))
    for g in groups:
        te = G == g
        pr = build(X[~te], isN2[~te], y[~te], G[~te])
        pred[te] = pr(X[te], isN2[te])
    oof[name] = pred

# ---- metrics ---------------------------------------------------------------------------
def metrics(pred, yy, mask=None, neutral=False):
    if mask is not None: pred, yy = pred[mask], yy[mask]
    if neutral:
        k = np.abs(yy) > 0.5; pred, yy = pred[k], yy[k]
    if len(yy) < 4: return dict(R2=np.nan, rho=np.nan, CCR=np.nan, RMSE=np.nan)
    r = np.corrcoef(pred, yy)[0, 1]
    nz = yy != 0
    return dict(R2=r * r if r > 0 else -(r * r),   # negative sign flags an inverted relationship
                rho=stats.spearmanr(pred, yy)[0],
                CCR=np.mean(np.sign(pred[nz]) == np.sign(yy[nz])),
                RMSE=np.sqrt(np.mean((pred - yy) ** 2)))

subsets = {'All rows': np.ones(len(y), bool),
           'NBD1': (df['domain'] == 'NBD1').values,
           'NBD2': (df['domain'] == 'NBD2').values}
subsets = {k: v for k, v in subsets.items() if v.sum() > 0}

def boot_ci(name, subset, key, B):
    idx_by_g = {g: np.where((G == g) & subsets[subset])[0] for g in groups}
    gl = [g for g in groups if len(idx_by_g[g])]
    vals = []
    for _ in range(B):
        pick = rng.choice(len(gl), len(gl))
        ii = np.concatenate([idx_by_g[gl[k]] for k in pick])
        m = metrics(oof[name][ii], y[ii]); vals.append(m[key])
    vals = np.array(vals, float)
    return np.nanpercentile(vals, [2.5, 97.5])

rows = []
for name in models:
    for sub, mask in subsets.items():
        m = metrics(oof[name], y, mask)
        mn = metrics(oof[name], y, mask, neutral=True)
        lo, hi = boot_ci(name, sub, 'R2', a.boot)
        rl, rh = boot_ci(name, sub, 'rho', a.boot)
        rows.append(dict(model=name, subset=sub, n=int(mask.sum()), R2=m['R2'], R2_lo=lo, R2_hi=hi,
                         rho=m['rho'], rho_lo=rl, rho_hi=rh, CCR=m['CCR'], CCR_no_neutral=mn['CCR'],
                         RMSE_degC=m['RMSE']))
res = pd.DataFrame(rows)
res.to_csv(f'{a.out}/results_cv.csv', index=False)

# ---- paired comparison vs best single predictor (by pooled CV R2) -----------------------
single = [n for n in models if n.endswith('(calibrated)')]
base = max(single, key=lambda n: metrics(oof[n], y)['R2'])
paired = []
for name in models:
    if name == base: continue
    for sub in subsets:
        idx_by_g = {g: np.where((G == g) & subsets[sub])[0] for g in groups}
        gl = [g for g in groups if len(idx_by_g[g])]
        d = []
        for _ in range(a.boot):
            pick = rng.choice(len(gl), len(gl))
            ii = np.concatenate([idx_by_g[gl[k]] for k in pick])
            d.append(metrics(oof[name][ii], y[ii])['rho'] - metrics(oof[base][ii], y[ii])['rho'])
        d = np.array(d, float)
        obs = metrics(oof[name], y, subsets[sub])['rho'] - metrics(oof[base], y, subsets[sub])['rho']
        paired.append(dict(model=name, vs=base, subset=sub, delta_rho=obs,
                           lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5)))
pd.DataFrame(paired).to_csv(f'{a.out}/paired_vs_best_single.csv', index=False)

# ---- sensitivity: drop the three extreme NBD1 destabilisers --------------------------------
ext = df['mutation'].isin(['L467P', 'R560T', 'V520F']).values
sens = []
for name in models:
    for lab, mk in (('with all', np.ones(len(y), bool)), ('without L467P/R560T/V520F', ~ext)):
        m = metrics(oof[name], y, mk)
        sens.append(dict(model=name, set=lab, n=int(mk.sum()), R2=m['R2'], rho=m['rho'], CCR=m['CCR']))
pd.DataFrame(sens).to_csv(f'{a.out}/sensitivity_outliers.csv', index=False)

# ---- raw (uncalibrated) per-domain reference, Bahia-style ------------------------------------
raw = []
for f in feats:
    for sub in [k for k in ('NBD1', 'NBD2') if k in subsets]:
        mk = subsets[sub]
        x = df[f].values[mk]; yy = y[mk]
        r = np.corrcoef(-x, yy)[0, 1]
        raw.append(dict(feature=f, subset=sub, n=int(mk.sum()), R2_raw=r * r,
                        rho_raw=stats.spearmanr(-x, yy)[0]))
pd.DataFrame(raw).to_csv(f'{a.out}/raw_per_domain.csv', index=False)

out = df[['id', 'domain', 'mutation'] + (['construct'] if 'construct' in df else []) + ['exp_dTm']].copy() if 'id' in df else df[['domain', 'mutation', 'exp_dTm']].copy()
for n, pr in oof.items(): out['oof: ' + n] = pr
out.to_csv(f'{a.out}/oof_predictions.csv', index=False)

# ---- figures ----------------------------------------------------------------------------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
SURF, INK, MUTE, BLUE, ORANGE, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#e6e5e1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTE, 'axes.labelcolor': INK,
                     'xtick.color': MUTE, 'ytick.color': MUTE, 'text.color': INK})

# Fig 1: CV spearman +- CI by model, per subset
fig, axs = plt.subplots(1, len(subsets), figsize=(4.2 * len(subsets), 3.9), sharex=True, facecolor=SURF, squeeze=False); axs = axs[0]
order = list(models)
for ax, sub in zip(axs, subsets):
    ax.set_facecolor(SURF)
    d = res[res.subset == sub].set_index('model').loc[order]
    ypos = np.arange(len(order))[::-1]
    for yp, (n, r) in zip(ypos, d.iterrows()):
        c = ORANGE if n in ('z-score mean consensus', 'ridge stack (all features)') else BLUE
        ax.plot([abs(r.rho_lo), abs(r.rho_hi)] if r.rho < 0 else [r.rho_lo, r.rho_hi], [yp, yp], color=c, lw=2, solid_capstyle='round')
        ax.plot(r.rho, yp, 'o', ms=8, color=c, mec=SURF, mew=2)
    ax.set_yticks(ypos); ax.set_yticklabels(order if ax is axs[0] else [], fontsize=9)
    ax.set_title(f'{sub} (n={int(d.n.iloc[0])})', fontsize=11, loc='left')
    ax.axvline(0, color=MUTE, lw=0.8); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.set_xlabel('Spearman ρ, leave-one-position-out CV')
fig.suptitle('Cross-validated rank correlation with experimental ΔTm (bars: 95% bootstrap CI over positions)', x=0.01, ha='left', fontsize=11)
fig.tight_layout(); fig.savefig(f'{a.out}/fig_cv_spearman.png', dpi=170, facecolor=SURF); plt.close(fig)

# Fig 2: OOF scatter, best single vs ridge stack (if present)
show = [base] + (['ridge stack (all features)'] if len(feats) > 1 else [])
fig, axs = plt.subplots(1, len(show), figsize=(5.4 * len(show), 4.6), facecolor=SURF)
axs = np.atleast_1d(axs)
for ax, name in zip(axs, show):
    ax.set_facecolor(SURF)
    for dom, c in (('NBD1', BLUE), ('NBD2', ORANGE)):
        mk = (df['domain'] == dom).values
        ax.scatter(y[mk], oof[name][mk], s=42, color=c, edgecolor=SURF, linewidth=1.5, label=dom, zorder=3)
    lim = [min(y.min(), oof[name].min()) - 1, max(y.max(), oof[name].max()) + 1]
    ax.plot(lim, lim, color=MUTE, lw=0.9, ls=(0, (4, 3))); ax.set_xlim(lim); ax.set_ylim(lim)
    for k in np.argsort(-np.abs(y - oof[name]))[:3]:
        ax.annotate(df['mutation'][k], (y[k], oof[name][k]), xytext=(5, 5), textcoords='offset points', fontsize=8, color=MUTE)
    m = metrics(oof[name], y)
    ax.set_title(f'{name}\nρ={m["rho"]:.2f}  R²={m["R2"]:.2f}  CCR={m["CCR"]:.0%}', fontsize=10, loc='left')
    ax.set_xlabel('experimental ΔTm (°C)'); ax.set_ylabel('out-of-fold predicted ΔTm (°C)')
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
fig.tight_layout(); fig.savefig(f'{a.out}/fig_oof_scatter.png', dpi=170, facecolor=SURF); plt.close(fig)

# ---- console summary --------------------------------------------------------------------------
pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.float_format', lambda v: f'{v:.2f}')
print('\nBest single predictor by pooled CV R2:', base)
print('\n== Leave-one-position-out CV ==')
print(res[['model', 'subset', 'n', 'R2', 'R2_lo', 'R2_hi', 'rho', 'rho_lo', 'rho_hi', 'CCR', 'CCR_no_neutral', 'RMSE_degC']].to_string(index=False))
print('\n== Paired: delta Spearman vs best single (95% CI) ==')
print(pd.DataFrame(paired).to_string(index=False))
print('\n== Sensitivity to three extreme NBD1 points ==')
print(pd.DataFrame(sens).to_string(index=False))
print('\n== Raw per-domain (uncalibrated; compare Bahia Table 4) ==')
print(pd.DataFrame(raw).to_string(index=False))
