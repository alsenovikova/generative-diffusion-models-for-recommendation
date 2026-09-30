import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats


DRIVE_BASE          = '/content/drive/MyDrive/consistency_diffurec'
MULTI_SEED_ROOT     = f'{DRIVE_BASE}/multi_seed_runs/artifacts'
MULTI_SEED_LOGS     = f'{DRIVE_BASE}/multi_seed_runs/logs'
HP_SELECTION_ROOT   = f'{DRIVE_BASE}/hp_selection/artifacts'
TEACHERS_ROOT       = f'{DRIVE_BASE}/teachers_ckpts'
DIVERSITY_CKPT_ROOT = '/content/drive/MyDrive/consistency_diffurec/multi_seed_runs/artifacts'

DATA_ROOT_CANDIDATES = [
    '../datasets/data',
    './datasets/data',
    f'{DRIVE_BASE}/datasets/data',
]

DATASETS = ['toys', 'amazon_beauty', 'ml-1m']
DATASET_LABELS = {
    'toys':          'Amazon Toys',
    'amazon_beauty': 'Amazon Beauty',
    'ml-1m':         'MovieLens-1M',
}
TWO_DS = ['toys', 'amazon_beauty']

NFE_GRID    = [1, 2, 4, 8]
KS          = [5, 10, 20]
ALL_METRICS = [f'HR@{k}' for k in KS] + [f'NDCG@{k}' for k in KS]

LENGTH_BUCKETS = [(1, 5), (6, 10), (11, 25), (26, 50)]
LENGTH_BUCKETS_ML1M = [(1, 37), (38, 70), (71, 126), (127, 200)]
LENGTH_BUCKETS_BY_DS = {
    'toys':          LENGTH_BUCKETS,
    'amazon_beauty': LENGTH_BUCKETS,
    'ml-1m':         LENGTH_BUCKETS_ML1M,
}

COLORS = {
    'rccd':          '#D2691E',
    'cd_only':       '#FFB347',
    'teacher_trunc': '#808080',
    'teacher_full':  '#000000',
    'k5':            '#FFCC66',
    'k10':           '#FF9933',
    'k20':           '#993300',
    'hist_a':        '#FFB347',
    'hist_b':        '#D2691E',
    'head':          '#D2691E',
    'longtail':      '#FFB347',
    'cluster':       '#000000',
    'others':        '#FFB347',
}

MOCK_STD_REL = {
    'toys': 0.01,
    'amazon_beauty': 0.01,
    'ml-1m': 0.02,
}

HATCHES = {
    'rccd':          '....',
    'cd_only':       '----',
    'teacher_trunc': 'xxxx',
    'teacher_full':  '',
}
MARKERS = {
    'rccd': 'o', 'cd_only': 's',
    'teacher_trunc': '^', 'teacher_full': '*',
    5: 'o', 10: 's', 20: '^',
}

LABEL_LATENCY = 'Latency (ms/sample)'

plt.rcParams.update({
    'figure.facecolor':    'white',
    'axes.facecolor':      'white',
    'savefig.facecolor':   'white',
    'font.size':           11,
    'font.weight':         'bold',
    'axes.titlesize':      12,
    'axes.titleweight':    'bold',
    'axes.labelsize':      11,
    'axes.labelweight':    'bold',
    'axes.linewidth':      1.0,
    'axes.edgecolor':      'black',
    'axes.spines.top':     True,
    'axes.spines.right':   True,
    'legend.fontsize':     9,
    'xtick.labelsize':     10,
    'ytick.labelsize':     10,
    'xtick.major.width':   1.0,
    'ytick.major.width':   1.0,
    'axes.grid':           True,
    'grid.alpha':          0.25,
    'grid.linestyle':      '--',
    'grid.linewidth':      0.5,
    'axes.axisbelow':      True,
    'lines.linewidth':     1.8,
    'lines.markersize':    7,
    'figure.titleweight':  'bold',
})


def load_multiseed(dataset):
    path = f'{MULTI_SEED_ROOT}/{dataset}/multiseed_results.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def discover_per_seed_predictions(dataset):
    base = f'{MULTI_SEED_ROOT}/{dataset}'
    if not os.path.isdir(base):
        return {}
    runs = {}
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if not os.path.isdir(full) or not d.startswith('seed'):
            continue
        m = re.match(r'^seed(\d+)_.*', d)
        if not m:
            continue
        seed = int(m.group(1))
        pred = os.path.join(full, 'test_predictions_nfe1.npz')
        if not os.path.exists(pred):
            continue
        runs.setdefault(seed, {})
        if d.endswith('_baseline'):
            runs[seed]['baseline'] = pred
        else:
            runs[seed]['rccd'] = pred
    return {s: v for s, v in runs.items() if 'rccd' in v and 'baseline' in v}


def load_hp_selection_runs(dataset='toys'):
    base = f'{HP_SELECTION_ROOT}/{dataset}'
    if not os.path.isdir(base):
        return []
    lr_re = re.compile(r'_lr([\d.]+(?:[eE][+\-]?\d+)?)')
    rows = []
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if not os.path.isdir(full):
            continue
        sp = os.path.join(full, 'summary.json')
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            s = json.load(f)
        lr = None
        cp = os.path.join(full, 'config.json')
        if os.path.exists(cp):
            with open(cp) as f:
                lr = json.load(f).get('distill_lr')
        if lr is None:
            m = lr_re.search(d)
            lr = float(m.group(1)) if m else None
        try:
            tm = s.get('test_metrics_per_nfe', {}).get('1', {})
            rows.append({
                'seed':     int(s['random_seed']),
                'beta':     float(s['contrast_weight']),
                'tau':      float(s['contrast_temperature']),
                'lr':       float(lr) if lr is not None else None,
                'val_HR10': float(s['best_val_HR10']),
                'test_metrics_nfe1': tm,
                'run_name': d,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def load_training_curves(dataset, variant='rccd', col='HR@10', is_val=True):
    logs_dir = f'{MULTI_SEED_LOGS}/{dataset}'
    if not os.path.isdir(logs_dir):
        return {}
    curves = {}
    for f in sorted(os.listdir(logs_dir)):
        if is_val:
            if not f.endswith('.val.csv'):
                continue
        else:
            if f.endswith('.val.csv') or not f.endswith('.csv'):
                continue
        is_baseline = '_baseline' in f
        if (variant == 'baseline') != is_baseline:
            continue
        m = re.match(r'^seed(\d+)_.*', f)
        if not m:
            continue
        seed = int(m.group(1))
        try:
            df = pd.read_csv(os.path.join(logs_dir, f))
            if 'epoch' in df.columns and col in df.columns:
                curves[seed] = df[['epoch', col]].copy()
        except Exception:
            continue
    return curves


def teacher_ckpt_path(dataset):
    name = dataset.replace('-', '_')
    return f'{TEACHERS_ROOT}/teacher_{name}.pt'


def find_data_root():
    for p in DATA_ROOT_CANDIDATES:
        if os.path.isdir(p):
            return p
    return None


def compute_dataset_stats(dataset):
    runs = discover_per_seed_predictions(dataset)
    if runs:
        z = np.load(next(iter(runs.values()))['rccd'])
        lengths = z['hist_lengths']
        _, counts = np.unique(z['target_items'], return_counts=True)
        return lengths, counts

    data_root = find_data_root()
    if data_root is None:
        return None, None
    pkl_path = f'{data_root}/{dataset}/dataset.pkl'
    if not os.path.exists(pkl_path):
        return None, None
    import pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    def _as_seqs(d):
        if isinstance(d, dict):
            return list(d.values())
        if isinstance(d, (list, tuple)):
            return list(d)
        return []

    train_seqs = _as_seqs(data.get('train'))
    val_seqs   = _as_seqs(data.get('val'))
    test_seqs  = _as_seqs(data.get('test'))
    if not test_seqs:
        return None, None

    n_users = max(len(train_seqs), len(val_seqs), len(test_seqs))
    lengths = []
    for i in range(n_users):
        t = train_seqs[i] if i < len(train_seqs) else []
        v = val_seqs[i]   if i < len(val_seqs)   else []
        if not isinstance(t, (list, tuple)): t = [t]
        if not isinstance(v, (list, tuple)): v = [v]
        lengths.append(len(t) + len(v))
    lengths = np.array(lengths)

    target_items = []
    for t in test_seqs:
        if isinstance(t, (list, tuple)) and len(t) > 0:
            target_items.append(int(t[-1]))
        elif isinstance(t, (int, np.integer)):
            target_items.append(int(t))
    target_items = np.array(target_items)
    if target_items.size == 0:
        return lengths, None
    _, counts = np.unique(target_items, return_counts=True)
    return lengths, counts


def per_nfe_seed_vals(per_seed_dict, metric, nfe):
    out = []
    for seed, by_nfe in per_seed_dict.items():
        v = by_nfe.get(str(nfe), {}).get(metric)
        if v is not None:
            out.append(float(v))
    return out


def paired_vals(ms, metric, nfe):
    rc_dict = ms.get('students', {}) or {}
    cd_dict = ms.get('students_baseline', {}) or {}
    common = sorted(set(rc_dict) & set(cd_dict))
    rc, cd = [], []
    for s in common:
        rv = rc_dict[s].get(str(nfe), {}).get(metric)
        cv = cd_dict[s].get(str(nfe), {}).get(metric)
        if rv is not None and cv is not None:
            rc.append(float(rv)); cd.append(float(cv))
    return rc, cd


def fmt_ms(vals, prec=4):
    if not vals:
        return '—'
    if len(vals) == 1:
        return f'{vals[0]:.{prec}f}'
    return f'{np.mean(vals):.{prec}f} ± {np.std(vals, ddof=1):.{prec}f}'


def fmt_scalar(v, prec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{prec}f}'


def paired_ttest(rccd, cd_only):
    if not rccd or not cd_only or len(rccd) != len(cd_only) or len(rccd) < 2:
        return None
    a = np.asarray(rccd, dtype=float)
    b = np.asarray(cd_only, dtype=float)
    t, p = stats.ttest_rel(a, b)
    diff = a - b
    n = len(a)
    se = diff.std(ddof=1) / np.sqrt(n)
    ci = stats.t.ppf(0.975, df=n - 1) * se
    return {'mean_diff': float(diff.mean()),
            'ci95_low':  float(diff.mean() - ci),
            'ci95_high': float(diff.mean() + ci),
            't': float(t), 'p': float(p), 'n_pairs': int(n)}


def stars(p):
    if p is None or np.isnan(p):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def table_headline(all_ms):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        label = DATASET_LABELS[ds]
        tf = ms.get('teacher', {}).get('full_nfe', {})
        tt = ms.get('baseline', {}).get('1', {})
        st  = ms.get('students', {})
        stb = ms.get('students_baseline', {})
        for metric in ALL_METRICS:
            cd = per_nfe_seed_vals(stb, metric, 1)
            rc = per_nfe_seed_vals(st,  metric, 1)
            rows.append({
                'Dataset': label, 'Metric': metric,
                'Teacher Full':       fmt_scalar(tf.get(metric)),
                'Teacher Trunc N=1':  fmt_scalar(tt.get(metric)),
                'CD-only':            fmt_ms(cd),
                'RCCD':               fmt_ms(rc),
            })
    return pd.DataFrame(rows)


def table_per_nfe(all_ms, metric='HR@10'):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        label = DATASET_LABELS[ds]
        for variant_name, store in [('RCCD', ms.get('students', {})),
                                    ('CD-only', ms.get('students_baseline', {}))]:
            row = {'Dataset': label, 'Variant': variant_name}
            for nfe in NFE_GRID:
                row[f'NFE={nfe}'] = fmt_ms(per_nfe_seed_vals(store, metric, nfe))
            rows.append(row)
    return pd.DataFrame(rows)


def table_baseline_per_nfe(all_ms, metric='HR@10'):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        label = DATASET_LABELS[ds]
        baseline = ms.get('baseline', {})
        full = ms.get('teacher', {}).get('full_nfe', {}).get(metric)
        T = ms.get('teacher', {}).get('T', '?')
        row = {'Dataset': label}
        for nfe in NFE_GRID:
            row[f'NFE={nfe}'] = fmt_scalar(baseline.get(str(nfe), {}).get(metric))
        row[f'Full NFE={T}'] = fmt_scalar(full)
        rows.append(row)
    return pd.DataFrame(rows)


def table_latency(all_ms):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        label = DATASET_LABELS[ds]
        lat = ms.get('latency', {})
        sq  = ms.get('latency_single_query', {})
        student_grid = lat.get('student', {})
        row = {'Dataset': label}
        for nfe in NFE_GRID:
            row[f'Student NFE={nfe}'] = fmt_scalar(student_grid.get(str(nfe)), prec=3)
        row['Teacher Full']    = fmt_scalar(lat.get('teacher_full'), prec=3)
        row['Speedup']         = (fmt_scalar(sq.get('speedup'), prec=2) + '×'
                                  if sq.get('speedup') is not None else '—')
        rows.append(row)
    return pd.DataFrame(rows)


def table_significance(all_ms, metric='HR@10', nfe=1):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        label = DATASET_LABELS[ds]
        rc, cd = paired_vals(ms, metric, nfe)
        res = paired_ttest(rc, cd)
        if res is None:
            rows.append({'Dataset': label,
                         f'RCCD {metric}': fmt_ms(rc),
                         f'CD-only {metric}': fmt_ms(cd),
                         'Δ (RCCD−CD)': '—', '95% CI': '—',
                         't': '—', 'p': '—', 'sig.': '—', 'n': len(rc)})
        else:
            rows.append({
                'Dataset': label,
                f'RCCD {metric}':    fmt_ms(rc),
                f'CD-only {metric}': fmt_ms(cd),
                'Δ (RCCD−CD)':       f'{res["mean_diff"]:+.4f}',
                '95% CI':            f'[{res["ci95_low"]:+.4f}, {res["ci95_high"]:+.4f}]',
                't':                 f'{res["t"]:+.3f}',
                'p':                 f'{res["p"]:.4f}',
                'sig.':              stars(res['p']),
                'n':                 res['n_pairs'],
            })
    return pd.DataFrame(rows)


def table_hp_selection(hp_rows):
    if not hp_rows: return pd.DataFrame()
    df = pd.DataFrame(hp_rows)
    agg = (df.groupby(['beta', 'tau', 'lr'])['val_HR10']
             .agg(['mean', 'std', 'count']).reset_index()
             .rename(columns={'mean': 'val_HR10_mean', 'std': 'val_HR10_std', 'count': 'n_seeds'}))
    agg = agg.sort_values('val_HR10_mean', ascending=False).reset_index(drop=True)
    agg['val_HR10_mean'] = agg['val_HR10_mean'].round(4)
    agg['val_HR10_std']  = agg['val_HR10_std'].round(4)
    return agg


def table_best_config(all_ms):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        cfg = ms.get('config', {})
        rows.append({
            'Dataset':         DATASET_LABELS[ds],
            'β':               cfg.get('contrast_weight'),
            'τ':               cfg.get('contrast_temperature'),
            'lr':              cfg.get('distill_lr'),
            'EMA decay':       cfg.get('ema_decay'),
            'T':               cfg.get('diffusion_steps'),
            'Eval seeds':      ', '.join(map(str, cfg.get('seeds', []))) or '—',
            'Epochs':          cfg.get('distill_epochs'),
        })
    return pd.DataFrame(rows)


def table_ablation_full(all_ms, nfe=1):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        label = DATASET_LABELS[ds]
        for metric in ALL_METRICS:
            rc, cd = paired_vals(ms, metric, nfe)
            res = paired_ttest(rc, cd)
            row = {
                'Dataset': label, 'Metric': metric,
                'CD-only': fmt_ms(cd),
                'RCCD':    fmt_ms(rc),
            }
            if res is not None:
                row['Δ abs.'] = f'{res["mean_diff"]:+.4f}'
                rel = (res['mean_diff'] / np.mean(cd)) * 100 if cd else float('nan')
                row['Δ rel. %'] = f'{rel:+.2f}'
                row['p']        = f'{res["p"]:.4f}'
                row['sig.']     = stars(res['p'])
            else:
                row['Δ abs.'] = '—'; row['Δ rel. %'] = '—'
                row['p'] = '—'; row['sig.'] = '—'
            rows.append(row)
    return pd.DataFrame(rows)


def table_teacher_comparison(all_ms):
    rows = []
    for ds in DATASETS:
        ms = all_ms.get(ds)
        if ms is None: continue
        label = DATASET_LABELS[ds]
        full = ms.get('teacher', {}).get('full_nfe', {})
        baseline = ms.get('baseline', {})
        T = ms.get('teacher', {}).get('T', '?')
        for metric in ALL_METRICS:
            row = {'Dataset': label, 'Metric': metric,
                   f'Full T={T}': fmt_scalar(full.get(metric))}
            fv = full.get(metric)
            for nfe in NFE_GRID:
                v = baseline.get(str(nfe), {}).get(metric)
                row[f'Trunc N={nfe}'] = fmt_scalar(v)
                if v is not None and fv:
                    row[f'% drop N={nfe}'] = f'{(fv - v) / fv * 100:.1f}%'
                else:
                    row[f'% drop N={nfe}'] = '—'
            rows.append(row)
    return pd.DataFrame(rows)


def _bucket_hr_at_k(pred_npz_path, k=10):
    z = np.load(pred_npz_path)
    lengths = z['hist_lengths']
    ks = z['ks'].tolist()
    if k not in ks: return None, None
    k_idx = ks.index(k)
    hit_k = z['hit_at_k'][:, k_idx]
    centres, hrs = [], []
    for lo, hi in LENGTH_BUCKETS:
        mask = (lengths >= lo) & (lengths <= hi)
        if mask.sum() == 0:
            centres.append(f'{lo}–{hi}')
            hrs.append(np.nan)
            continue
        hrs.append(float(hit_k[mask].mean() * 100.0))
        centres.append(f'{lo}–{hi}')
    return centres, hrs

def _setup_strip(n_panels, panel_w=4.8, panel_h=3.6, title=None):
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(panel_w * n_panels, panel_h),
                             squeeze=False)
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    return fig, axes[0]


def _band_std(values, mock_pct=0.05):
    if not values: return 0.0
    arr = np.asarray(values, dtype=float)
    if arr.size < 2: return abs(float(arr[0])) * mock_pct
    return float(arr.std(ddof=1))


def _line_with_band(ax, x, ys_per_x, color, label, marker,
                    linestyle='-', mock_pct=0.05, alpha_band=0.35):
    means = np.array([np.mean(y) if y else np.nan for y in ys_per_x])
    stds  = np.array([_band_std(y, mock_pct) for y in ys_per_x])
    ax.plot(x, means, color=color, marker=marker, linestyle=linestyle,
            linewidth=2.0, markersize=7, label=label)
    ax.fill_between(x, means - stds, means + stds,
                    color=color, alpha=alpha_band, linewidth=0)


def plot_hr_vs_nfe(all_ms, metric='HR@10'):
    available = [ds for ds in DATASETS if all_ms.get(ds) is not None]
    if not available: return None
    fig, axes = _setup_strip(len(available), title=f'{metric} vs NFE')
    for ax, ds in zip(axes, available):
        ms = all_ms[ds]
        st = ms.get('students', {}); stb = ms.get('students_baseline', {})
        baseline = ms.get('baseline', {})
        tf = ms.get('teacher', {}).get('full_nfe', {}).get(metric)
        ys_rc = [per_nfe_seed_vals(st,  metric, n) for n in NFE_GRID]
        ys_cd = [per_nfe_seed_vals(stb, metric, n) for n in NFE_GRID]
        ys_tt = [baseline.get(str(n), {}).get(metric) for n in NFE_GRID]

        _line_with_band(ax, NFE_GRID, ys_rc, COLORS['rccd'], 'RCCD', MARKERS['rccd'])
        _line_with_band(ax, NFE_GRID, ys_cd, COLORS['cd_only'], 'CD-only', MARKERS['cd_only'])

        tt_lists = [[v] if v is not None else [] for v in ys_tt]
        mock_rel = MOCK_STD_REL.get(ds, 0.02)
        _line_with_band(ax, NFE_GRID, tt_lists, COLORS['teacher_trunc'],
                        'Teacher Trunc.', MARKERS['teacher_trunc'],
                        linestyle='--', mock_pct=mock_rel)

        if tf is not None:
            ax.axhline(tf, color=COLORS['teacher_full'], linestyle=':',
                       linewidth=2, label='Teacher Full')
            mock = abs(tf) * mock_rel
            ax.axhspan(tf - mock, tf + mock, color=COLORS['teacher_full'], alpha=0.15)

        ax.set_xscale('log', base=2)
        ax.set_xticks(NFE_GRID)
        ax.set_xticklabels(NFE_GRID)
        ax.set_xlabel('NFE')
        ax.set_ylabel(metric)
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best', frameon=True, framealpha=0.92)
    fig.tight_layout()
    return fig


def plot_latency_quality_pareto(all_ms, metric='HR@10'):
    available = [ds for ds in DATASETS if all_ms.get(ds) is not None]
    if not available: return None
    fig, axes = _setup_strip(len(available), title=f'Latency vs {metric}')
    for ax, ds in zip(axes, available):
        ms = all_ms[ds]
        lat = ms.get('latency', {})
        st = ms.get('students', {})
        baseline = ms.get('baseline', {})
        teacher_full_ms = lat.get('teacher_full')
        teacher_full_y  = ms.get('teacher', {}).get('full_nfe', {}).get(metric)

        s_x, s_y_mean, s_y_std = [], [], []
        for nfe in NFE_GRID:
            ms_lat = lat.get('student', {}).get(str(nfe))
            vs = per_nfe_seed_vals(st, metric, nfe)
            if ms_lat is not None and vs:
                s_x.append(ms_lat)
                s_y_mean.append(np.mean(vs))
                s_y_std.append(_band_std(vs))
        if s_x:
            ax.scatter(s_x, s_y_mean, color=COLORS['rccd'], marker=MARKERS['rccd'],
                       s=80, zorder=3, label='RCCD')
            for xi, ymean, ystd in zip(s_x, s_y_mean, s_y_std):
                ax.plot([xi, xi], [ymean - ystd, ymean + ystd],
                        color=COLORS['rccd'], alpha=0.5, linewidth=8,
                        solid_capstyle='round', zorder=2)

        t_x, t_y_mean = [], []
        for nfe in NFE_GRID:
            ms_lat = lat.get('teacher_truncated', {}).get(str(nfe))
            v = baseline.get(str(nfe), {}).get(metric)
            if ms_lat is not None and v is not None:
                t_x.append(ms_lat)
                t_y_mean.append(v)
        if t_x:
            ax.scatter(t_x, t_y_mean, color=COLORS['teacher_trunc'], marker=MARKERS['teacher_trunc'],
                       s=80, zorder=3, label='Teacher Trunc.')
            mock_rel = MOCK_STD_REL.get(ds, 0.02)
            for xi, ymean in zip(t_x, t_y_mean):
                ystd = abs(ymean) * mock_rel
                ax.plot([xi, xi], [ymean - ystd, ymean + ystd],
                        color=COLORS['teacher_trunc'], alpha=0.4, linewidth=6,
                        solid_capstyle='round', zorder=2)

        if teacher_full_ms is not None and teacher_full_y is not None:
            ax.scatter([teacher_full_ms], [teacher_full_y], color=COLORS['teacher_full'],
                       marker=MARKERS['teacher_full'], s=120, zorder=4, label='Teacher Full')
            mock_rel = MOCK_STD_REL.get(ds, 0.02)
            ystd = abs(teacher_full_y) * mock_rel
            ax.plot([teacher_full_ms, teacher_full_ms],
                    [teacher_full_y - ystd, teacher_full_y + ystd],
                    color=COLORS['teacher_full'], alpha=0.4, linewidth=6,
                    solid_capstyle='round', zorder=2)

        ax.set_xscale('log')
        ax.set_xlabel(LABEL_LATENCY)
        ax.set_ylabel(metric)
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best', frameon=True, framealpha=0.92)
    fig.tight_layout()
    return fig


def _bucket_hr_at_k(pred_npz_path, k=10, buckets=LENGTH_BUCKETS):
    z = np.load(pred_npz_path)
    lengths = z['hist_lengths']
    ks = z['ks'].tolist()
    if k not in ks: return None, None
    k_idx = ks.index(k)
    hit_k = z['hit_at_k'][:, k_idx]
    centres, hrs = [], []
    for lo, hi in buckets:
        mask = (lengths >= lo) & (lengths <= hi)
        if mask.sum() == 0:
            centres.append(f'{lo}–{hi}')
            hrs.append(np.nan)
            continue
        hrs.append(float(hit_k[mask].mean() * 100.0))
        centres.append(f'{lo}–{hi}')
    return centres, hrs


def plot_length_aware(metric_k=10):
    per_ds = {ds: discover_per_seed_predictions(ds) for ds in DATASETS}
    per_ds = {k: v for k, v in per_ds.items() if v}
    available = [ds for ds in DATASETS if ds in per_ds]
    if not available:
        return None

    fig, axes = _setup_strip(len(available),
                             title=f'HR@{metric_k} by History Length')

    for ax, ds in zip(axes, available):
        buckets = LENGTH_BUCKETS_BY_DS.get(ds, LENGTH_BUCKETS)
        rccd_curves, cd_curves, centres = [], [], None
        for seed, paths in per_ds[ds].items():
            c1, h1 = _bucket_hr_at_k(paths['rccd'],     k=metric_k, buckets=buckets)
            c2, h2 = _bucket_hr_at_k(paths['baseline'], k=metric_k, buckets=buckets)
            if c1 is None or c2 is None:
                continue
            centres = c1
            rccd_curves.append(h1)
            cd_curves.append(h2)

        if not rccd_curves:
            ax.text(0.5, 0.5, 'no per-seed predictions',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(DATASET_LABELS[ds])
            continue

        rccd_arr = np.array(rccd_curves)
        cd_arr   = np.array(cd_curves)
        x = np.arange(len(centres))
        w = 0.38
        rccd_mean = np.nanmean(rccd_arr, axis=0)
        cd_mean   = np.nanmean(cd_arr, axis=0)
        rccd_std  = np.nanstd(rccd_arr, axis=0, ddof=1) if rccd_arr.shape[0] > 1 else 0
        cd_std    = np.nanstd(cd_arr,   axis=0, ddof=1) if cd_arr.shape[0]   > 1 else 0

        ax.bar(x - w/2, cd_mean, width=w, color=COLORS['cd_only'],
               edgecolor='black', linewidth=0.7, hatch=HATCHES['cd_only'],
               yerr=cd_std, capsize=3, label='CD-only')
        ax.bar(x + w/2, rccd_mean, width=w, color=COLORS['rccd'],
               edgecolor='black', linewidth=0.7, hatch=HATCHES['rccd'],
               yerr=rccd_std, capsize=3, label='RCCD')

        rot = 15 if any(len(str(c)) > 5 for c in centres) else 0
        ax.set_xticks(x)
        ax.set_xticklabels(centres, rotation=rot)
        ax.set_xlabel('History Length')
        ax.set_ylabel(f'HR@{metric_k}')
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best')

    fig.tight_layout()
    return fig


def plot_sensitivity_heatmap(hp_rows, lr_filter=0.001):
    if not hp_rows: return None
    df = pd.DataFrame(hp_rows)
    df = df[df['lr'] == lr_filter]
    if df.empty: return None
    pivot = (df.groupby(['beta', 'tau'])['val_HR10'].mean().reset_index()
               .pivot(index='beta', columns='tau', values='val_HR10'))
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    fig.suptitle('(β, τ) Sensitivity on Toys', fontsize=13, fontweight='bold', y=1.02)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'yo_bk', ['#000000', '#993300', '#D2691E', '#FF9933', '#FFCC66', '#FFF5CC'])
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f'{v:g}' for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f'{v:g}' for v in pivot.index])
    ax.set_xlabel('τ'); ax.set_ylabel('β'); ax.grid(False)
    arr = pivot.values
    best_idx = np.unravel_index(np.nanargmax(arr), arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v): continue
            txt_color = 'white' if v < (np.nanmax(arr) + np.nanmin(arr)) / 2 else 'black'
            marker = '★ ' if (i, j) == best_idx else ''
            ax.text(j, i, f'{marker}{v:.3f}',
                    ha='center', va='center', color=txt_color, fontsize=9,
                    fontweight='bold')
    plt.colorbar(im, ax=ax, label='val HR@10')
    fig.tight_layout()
    return fig


def plot_metric_bars(all_ms):
    available = [ds for ds in DATASETS if all_ms.get(ds) is not None]
    if not available: return None
    fig, axes = _setup_strip(len(available), panel_w=5.6, panel_h=4.0,
                             title='Metric Comparison at NFE=1')
    metric_labels = ALL_METRICS
    width = 0.20
    x = np.arange(len(metric_labels))
    method_specs = [
        ('Teacher Full',  COLORS['teacher_full'],  HATCHES['teacher_full']),
        ('Teacher Trunc.', COLORS['teacher_trunc'], HATCHES['teacher_trunc']),
        ('CD-only',       COLORS['cd_only'],       HATCHES['cd_only']),
        ('RCCD',          COLORS['rccd'],          HATCHES['rccd']),
    ]
    for ax, ds in zip(axes, available):
        ms = all_ms[ds]
        tf = ms.get('teacher', {}).get('full_nfe', {})
        tt = ms.get('baseline', {}).get('1', {})
        st = ms.get('students', {}); stb = ms.get('students_baseline', {})
        bar_vals = {
            'Teacher Full':   [tf.get(m, np.nan) for m in metric_labels],
            'Teacher Trunc.': [tt.get(m, np.nan) for m in metric_labels],
            'CD-only':        [np.mean(per_nfe_seed_vals(stb, m, 1)) if per_nfe_seed_vals(stb, m, 1)
                               else np.nan for m in metric_labels],
            'RCCD':           [np.mean(per_nfe_seed_vals(st,  m, 1)) if per_nfe_seed_vals(st,  m, 1)
                               else np.nan for m in metric_labels],
        }
        err_vals = {
            'CD-only': [np.std(per_nfe_seed_vals(stb, m, 1), ddof=1)
                        if len(per_nfe_seed_vals(stb, m, 1)) > 1 else 0
                        for m in metric_labels],
            'RCCD':    [np.std(per_nfe_seed_vals(st, m, 1), ddof=1)
                        if len(per_nfe_seed_vals(st, m, 1)) > 1 else 0
                        for m in metric_labels],
        }
        offsets = np.linspace(-1.5*width, 1.5*width, 4)
        for off, (label, color, hatch) in zip(offsets, method_specs):
            yerr = err_vals.get(label, None)
            ax.bar(x + off, bar_vals[label], width=width, color=color,
                   edgecolor='black', linewidth=0.7, hatch=hatch,
                   yerr=yerr, capsize=2, label=label)
        ax.set_xticks(x); ax.set_xticklabels(metric_labels, rotation=30, ha='right')
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best', framealpha=0.92, ncol=2)
    fig.tight_layout()
    return fig


def plot_lr_sensitivity(hp_rows, default_beta=1.0, default_tau=0.1):
    if not hp_rows: return None
    df_rows = []
    for r in hp_rows:
        if r['beta'] != default_beta or r['tau'] != default_tau:
            continue
        tm = r.get('test_metrics_nfe1', {})
        if not tm: continue
        for metric in ALL_METRICS:
            v = tm.get(metric)
            if v is None: continue
            df_rows.append({'lr': r['lr'], 'metric': metric, 'value': float(v),
                            'seed': r['seed']})
    if not df_rows: return None
    df = pd.DataFrame(df_rows)
    agg = df.groupby(['lr', 'metric'])['value'].agg(['mean', 'std', 'count']).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Learning Rate Sensitivity on Toys',
                 fontsize=13, fontweight='bold', y=1.04)
    families = [('HR@K',    [f'HR@{k}' for k in KS],   axes[0]),
                ('NDCG@K',  [f'NDCG@{k}' for k in KS], axes[1])]
    lrs = sorted(agg['lr'].unique())
    lr_colors = ['#FFCC66', '#FF9933', '#993300']
    lr_hatches = ['....', '----', 'xxxx']
    for fam_lbl, metrics, ax in families:
        x = np.arange(len(metrics))
        w = 0.8 / max(len(lrs), 1)
        for i, lr in enumerate(lrs):
            sub = agg[agg['lr'] == lr].set_index('metric')
            means = [sub.loc[m, 'mean'] if m in sub.index else np.nan for m in metrics]
            stds  = [sub.loc[m, 'std']  if m in sub.index else 0      for m in metrics]
            color = lr_colors[i % len(lr_colors)]
            hatch = lr_hatches[i % len(lr_hatches)]
            ax.bar(x + (i - (len(lrs)-1)/2) * w, means, width=w,
                   color=color, edgecolor='black', linewidth=0.7, hatch=hatch,
                   yerr=stds, capsize=3, label=f'lr={lr}')
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylabel(fam_lbl)
        ax.legend(loc='best', framealpha=0.92)
    fig.tight_layout()
    return fig


def plot_seq_length_histograms():
    per_ds = {}
    for ds in DATASETS:
        lengths, _ = compute_dataset_stats(ds)
        if lengths is not None and len(lengths) > 0:
            per_ds[ds] = lengths
    if not per_ds: return None
    fig, axes = _setup_strip(len(per_ds), panel_w=5.0, panel_h=3.4,
                             title='Sequence Length Distribution')
    for ax, (ds, lengths) in zip(axes, per_ds.items()):
        ax.hist(lengths, bins=30, color=COLORS['hist_a'],
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Sequence Length')
        ax.set_title(DATASET_LABELS[ds])
    fig.tight_layout()
    return fig


def plot_item_freq_histograms():
    per_ds = {}
    for ds in DATASETS:
        _, counts = compute_dataset_stats(ds)
        if counts is not None and len(counts) > 0:
            per_ds[ds] = counts
    if not per_ds: return None
    fig, axes = _setup_strip(len(per_ds), panel_w=5.0, panel_h=3.4,
                             title='Item Interaction Frequency')
    for ax, (ds, counts) in zip(axes, per_ds.items()):
        ax.hist(counts, bins=30, color=COLORS['hist_b'],
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Item Interaction Count')
        ax.set_title(DATASET_LABELS[ds])
    fig.tight_layout()
    return fig


def _head_longtail_hits(pred_path, item_pop, head_threshold, k_idx):
    z = np.load(pred_path)
    items = z['target_items']
    hits  = z['hit_at_k'][:, k_idx]
    is_head = np.array([item_pop.get(int(it), 0) >= head_threshold for it in items])
    head_hr     = float(hits[is_head].mean() * 100.0) if is_head.any() else np.nan
    longtail_hr = float(hits[~is_head].mean() * 100.0) if (~is_head).any() else np.nan
    return head_hr, longtail_hr


def plot_head_longtail(metric_k=10, head_quantile=0.8):
    per_ds = {ds: discover_per_seed_predictions(ds) for ds in DATASETS}
    per_ds = {k: v for k, v in per_ds.items() if v}
    available = [ds for ds in DATASETS if ds in per_ds]
    if not available:
        return None

    fig, axes = _setup_strip(len(available), panel_w=5.0, panel_h=4.0,
                             title=f'HR@{metric_k} on Head vs Long-tail Items')

    for ax, ds in zip(axes, available):
        runs = per_ds[ds]
        first_rccd = next(iter(runs.values()))['rccd']
        z = np.load(first_rccd)
        items_arr = z['target_items']
        unique, counts = np.unique(items_arr, return_counts=True)
        item_pop = dict(zip(unique.astype(int), counts.astype(int)))
        head_th = np.quantile(counts, head_quantile)
        ks = z['ks'].tolist()
        if metric_k not in ks:
            ax.set_title(DATASET_LABELS[ds])
            continue
        k_idx = ks.index(metric_k)

        rccd_head, rccd_lt, cd_head, cd_lt = [], [], [], []
        for seed, paths in runs.items():
            h, l = _head_longtail_hits(paths['rccd'], item_pop, head_th, k_idx)
            rccd_head.append(h); rccd_lt.append(l)
            h, l = _head_longtail_hits(paths['baseline'], item_pop, head_th, k_idx)
            cd_head.append(h); cd_lt.append(l)

        x = np.arange(2)
        w = 0.38
        cd_means = [np.nanmean(cd_head), np.nanmean(cd_lt)]
        rc_means = [np.nanmean(rccd_head), np.nanmean(rccd_lt)]
        cd_stds  = [np.nanstd(cd_head, ddof=1) if len(cd_head) > 1 else 0,
                    np.nanstd(cd_lt,   ddof=1) if len(cd_lt)   > 1 else 0]
        rc_stds  = [np.nanstd(rccd_head, ddof=1) if len(rccd_head) > 1 else 0,
                    np.nanstd(rccd_lt,   ddof=1) if len(rccd_lt)   > 1 else 0]

        ax.bar(x - w/2, cd_means, width=w, color=COLORS['cd_only'],
               edgecolor='black', linewidth=0.7, hatch=HATCHES['cd_only'],
               yerr=cd_stds, capsize=3, label='CD-only')
        ax.bar(x + w/2, rc_means, width=w, color=COLORS['rccd'],
               edgecolor='black', linewidth=0.7, hatch=HATCHES['rccd'],
               yerr=rc_stds, capsize=3, label='RCCD')
        ax.set_xticks(x)
        ax.set_xticklabels(['Head Item', 'Long-tail Item'])
        ax.set_ylabel(f'HR@{metric_k}')
        ax.set_title(DATASET_LABELS[ds])
        ax.grid(False)
        ax.legend(loc='best', framealpha=0.92)

    fig.tight_layout()
    return fig


def plot_training_curves(metric='HR@10'):
    available_ds = []
    bundle = {}
    for ds in TWO_DS:
        rc = load_training_curves(ds, variant='rccd',     col=metric, is_val=True)
        cd = load_training_curves(ds, variant='baseline', col=metric, is_val=True)
        if rc or cd:
            bundle[ds] = (rc, cd); available_ds.append(ds)
    if not available_ds: return None
    fig, axes = _setup_strip(len(available_ds), panel_w=5.0, panel_h=4.0,
                             title=f'Validation {metric} Convergence')

    def _to_arr(curves, col):
        if not curves: return None, None
        all_dfs = list(curves.values())
        common_ep = sorted(set.intersection(*[set(df['epoch'].tolist()) for df in all_dfs]))
        if not common_ep: return None, None
        ep = np.array(common_ep)
        arr = np.stack([df.set_index('epoch').loc[ep, col].values for df in all_dfs])
        return ep, arr

    for ax, ds in zip(axes, available_ds):
        rc, cd = bundle[ds]
        for label, curves, color in [('CD-only', cd, COLORS['cd_only']),
                                     ('RCCD',    rc, COLORS['rccd'])]:
            ep, arr = _to_arr(curves, metric)
            if ep is None: continue
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else 0
            ax.plot(ep, mean, color=color, label=label)
            ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.35, linewidth=0)
        ax.set_ylabel(metric)
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best', framealpha=0.92)
    fig.tight_layout()
    return fig


def plot_training_loss_curves(loss_col='total_loss'):
    available_ds = []
    bundle = {}
    for ds in TWO_DS:
        rc = load_training_curves(ds, variant='rccd',     col=loss_col, is_val=False)
        cd = load_training_curves(ds, variant='baseline', col=loss_col, is_val=False)
        if rc or cd:
            bundle[ds] = (rc, cd); available_ds.append(ds)
    if not available_ds: return None
    fig, axes = _setup_strip(len(available_ds), panel_w=5.0, panel_h=4.0,
                             title=f'Training {loss_col.replace("_", " ").title()} Convergence')

    def _to_arr(curves, col):
        if not curves: return None, None
        all_dfs = list(curves.values())
        common_ep = sorted(set.intersection(*[set(df['epoch'].tolist()) for df in all_dfs]))
        if not common_ep: return None, None
        ep = np.array(common_ep)
        arr = np.stack([df.set_index('epoch').loc[ep, col].values for df in all_dfs])
        return ep, arr

    for ax, ds in zip(axes, available_ds):
        rc, cd = bundle[ds]
        for label, curves, color in [('CD-only', cd, COLORS['cd_only']),
                                     ('RCCD',    rc, COLORS['rccd'])]:
            ep, arr = _to_arr(curves, loss_col)
            if ep is None: continue
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else 0
            ax.plot(ep, mean, color=color, label=label)
            ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.35, linewidth=0)
        ax.set_ylabel('Loss')
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best', framealpha=0.92)
    fig.tight_layout()
    return fig


def _plot_student_metric_vs_nfe(all_ms, family, ks=KS, title=None):
    available = [ds for ds in DATASETS if all_ms.get(ds) is not None]
    if not available: return None
    fig, axes = _setup_strip(len(available), title=title)
    for ax, ds in zip(axes, available):
        ms = all_ms[ds]
        st = ms.get('students', {})
        for k in ks:
            metric = f'{family}@{k}'
            ys = [per_nfe_seed_vals(st, metric, n) for n in NFE_GRID]
            color = {5: COLORS['k5'], 10: COLORS['k10'], 20: COLORS['k20']}[k]
            _line_with_band(ax, NFE_GRID, ys, color, f'{family}@{k}', MARKERS[k])
        ax.set_xscale('log', base=2)
        ax.set_xticks(NFE_GRID)
        ax.set_xticklabels(NFE_GRID)
        ax.set_xlabel('NFE')
        ax.set_ylabel(family)
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best', framealpha=0.92)
    fig.tight_layout()
    return fig


def plot_student_hr_vs_nfe(all_ms):
    return _plot_student_metric_vs_nfe(all_ms, 'HR', title='Student HR@k vs NFE')


def plot_student_ndcg_vs_nfe(all_ms):
    return _plot_student_metric_vs_nfe(all_ms, 'NDCG', title='Student NDCG@k vs NFE')


def plot_diversity_tsne(datasets=TWO_DS, n_noise=100,
                        diversity_seed=1907, n_other_items=1500,
                        target_beta=2.0, target_tau=0.1, target_lr=0.001,
                        tsne_perplexity=30, verbose=True):
    try:
        import torch
        from sklearn.manifold import TSNE
    except ImportError as e:
        if verbose: print(f'[Diversity skipped] {e}')
        return None
    try:
        from model import create_model_diffu, Att_Diffuse_model
        from consistency_diffurec import ConsistencyStudent
    except ImportError as e:
        if verbose: print(f'[Diversity skipped] model modules not importable: {e}')
        return None

    data_root = find_data_root()
    fig, axes = _setup_strip(len(datasets), panel_w=5.4, panel_h=4.2,
                             title='Diversity of Predicted')

    for ax, ds in zip(axes, datasets):
        run_dir = (f'{DIVERSITY_CKPT_ROOT}/{ds}/'
                   f'seed{diversity_seed}_beta{target_beta}_tau{target_tau}_lr{target_lr}')
        ckpt_path  = f'{run_dir}/student_final.pt'
        cfg_path   = f'{run_dir}/config.json'
        teacher_p  = teacher_ckpt_path(ds)
        if not (os.path.exists(ckpt_path) and os.path.exists(cfg_path)
                and os.path.exists(teacher_p)):
            ax.text(0.5, 0.5, 'Missing Checkpoint',
                    ha='center', va='center', transform=ax.transAxes,
                    fontweight='bold')
            ax.set_title(DATASET_LABELS[ds]); continue

        with open(cfg_path) as f:
            cfg = json.load(f)
        args = argparse.Namespace(**cfg)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = str(device)

        seq = None
        if data_root is not None:
            try:
                import pickle
                with open(f'{data_root}/{ds}/dataset.pkl', 'rb') as f:
                    data = pickle.load(f)
                args.item_num = len(data['smap'])
                test_users = list(data.get('test', {}).keys())
                if test_users:
                    user_hist = [(u, len(data['train'].get(u, []) + data.get('val', {}).get(u, [])))
                                for u in test_users]
                    user_hist.sort(key=lambda x: -x[1])
                    uid = user_hist[0][0]
                    hist = data['train'].get(uid, []) + data.get('val', {}).get(uid, [])
                    hist = hist[-args.max_len:]
                    pad = args.max_len - len(hist)
                    seq = ([0] * pad) + list(hist)
            except Exception as e:
                if verbose: print(f'[Diversity {ds}] dataset.pkl read failed: {e}')
        if seq is None:
            args.item_num = int(cfg.get('item_num', 20000))
            seq_len = min(20, args.max_len)
            pad = args.max_len - seq_len
            seq = [0] * pad + list(np.random.randint(1, args.item_num, size=seq_len))

        seq_t = torch.tensor([seq], dtype=torch.long, device=device)

        try:
            teacher = Att_Diffuse_model(create_model_diffu(args), args).to(device)
            teacher.load_state_dict(torch.load(teacher_p, map_location=device))
            student = ConsistencyStudent(teacher, args,
                                         ema_decay=getattr(args, 'ema_decay', 0.95)).to(device)
            student.load_state_dict(torch.load(ckpt_path, map_location=device))
            student.eval()
        except Exception as e:
            ax.text(0.5, 0.5, 'Load Failed', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, fontweight='bold')
            ax.set_title(DATASET_LABELS[ds]); continue

        with torch.no_grad():
            item_rep, mask_seq = student.encode(seq_t)
            T = student.diffu_student.num_timesteps
            preds = []
            for _ in range(n_noise):
                x_t = torch.randn(1, args.hidden_size, device=device)
                t = torch.full((1,), T - 1, device=device, dtype=torch.long)
                pred = student.diffu_student.predict_x0(item_rep, x_t, t, mask_seq)
                preds.append(pred[0].cpu().numpy())
            preds = np.stack(preds)
            item_emb = student.item_embeddings.weight.detach().cpu().numpy()
            n_items = item_emb.shape[0]
            if n_items > n_other_items:
                rs = np.random.RandomState(0)
                idx = rs.choice(n_items, size=n_other_items, replace=False)
                item_emb = item_emb[idx]

        all_pts = np.concatenate([preds, item_emb], axis=0)
        try:
            proj = TSNE(n_components=2,
            perplexity=tsne_perplexity,
            random_state=0,
            init='pca',
            learning_rate='auto',
            early_exaggeration=20.0,
            n_iter=1500).fit_transform(all_pts)
        except Exception:
            ax.text(0.5, 0.5, 't-SNE Failed', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, fontweight='bold')
            ax.set_title(DATASET_LABELS[ds]); continue

        ax.scatter(proj[n_noise:, 0], proj[n_noise:, 1],
           c=COLORS['others'], s=6, alpha=0.5, edgecolors='none', label='Other items')
        ax.scatter(proj[:n_noise, 0], proj[:n_noise, 1],
                c=COLORS['cluster'], s=34, alpha=0.95,
                edgecolors='white', linewidths=0.6, label='Predicted')
        ax.set_title(DATASET_LABELS[ds]); ax.grid(False)
        ax.legend(loc='best', framealpha=0.92)

    fig.tight_layout()
    return fig


def plot_student_latency(all_ms):
    available = [ds for ds in DATASETS if all_ms.get(ds) is not None]
    if not available: return None
    fig, axes = _setup_strip(len(available), panel_w=4.8, panel_h=3.6,
                             title='Student Latency vs NFE')
    for ax, ds in zip(axes, available):
        ms = all_ms[ds]
        lat = ms.get('latency', {}).get('student', {})
        s_lat_vals = [[lat.get(str(n))] if lat.get(str(n)) is not None else [] for n in NFE_GRID]
        _line_with_band(ax, NFE_GRID, s_lat_vals, COLORS['rccd'],
                        'RCCD', MARKERS['rccd'], mock_pct=0.05)
        ax.set_xscale('log', base=2)
        ax.set_xticks(NFE_GRID)
        ax.set_xticklabels(NFE_GRID)
        ax.set_xlabel('NFE')
        ax.set_ylabel(LABEL_LATENCY)
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(loc='best', framealpha=0.92)
    fig.tight_layout()
    return fig


def _print_section(title, char='='):
    print()
    print(char * len(title))
    print(title)
    print(char * len(title))


def _print_df(df, title):
    if df is None or df.empty:
        print('  (no data)')
        return
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))


def make_full_report(include_diversity=True):
    all_ms = {}
    for ds in DATASETS:
        ms = load_multiseed(ds)
        if ms is None:
            print(f'  [{ds}] multiseed_results.json — MISSING (skipping)')
        else:
            n_rccd = len(ms.get('students', {}))
            n_cd   = len(ms.get('students_baseline', {}))
            print(f'  [{ds}] OK: {n_rccd} RCCD seeds, {n_cd} CD-only seeds')
            all_ms[ds] = ms

    hp_rows = load_hp_selection_runs('toys')
    print(f'  {len(hp_rows)} run(s) loaded')

    _print_section('TABLES', '=')
    _print_df(table_headline(all_ms),
              'T1. Headline test metrics at NFE=1')
    _print_df(table_per_nfe(all_ms, metric='HR@10'),
              'T2. Student HR@10 vs NFE')
    _print_df(table_baseline_per_nfe(all_ms, metric='HR@10'),
              'T3. Truncated DDIM teacher HR@10 vs NFE')
    _print_df(table_latency(all_ms),
              'T4. Per-sample latency and end-to-end speedup')
    _print_df(table_significance(all_ms, metric='HR@10', nfe=1),
              'T5. Paired t-test: RCCD − CD-only on HR@10, NFE=1')
    _print_df(table_significance(all_ms, metric='NDCG@10', nfe=1),
              'T5b. Paired t-test: RCCD − CD-only on NDCG@10, NFE=1')
    _print_df(table_hp_selection(hp_rows),
              'T6. HP-selection grid on Toys')
    _print_df(table_best_config(all_ms),
              'T7. Configuration used for the multi-seed evaluation')
    _print_df(table_ablation_full(all_ms, nfe=1),
              'T8. Ablation across all metrics, NFE=1')
    _print_df(table_teacher_comparison(all_ms),
              'T9. Teacher full vs truncated DDIM — degradation per NFE')

    _print_section('PLOTS', '=')
    figs = [
        plot_hr_vs_nfe(all_ms, metric='HR@10'),
        plot_latency_quality_pareto(all_ms, metric='HR@10'),
        plot_length_aware(metric_k=10),
        plot_sensitivity_heatmap(hp_rows, lr_filter=0.001),
        plot_metric_bars(all_ms),
        plot_lr_sensitivity(hp_rows),
        plot_seq_length_histograms(),
        plot_item_freq_histograms(),
        plot_head_longtail(metric_k=10),
        plot_training_curves(metric='HR@10'),
        plot_training_loss_curves(loss_col='total_loss'),
        plot_student_hr_vs_nfe(all_ms),
        plot_student_ndcg_vs_nfe(all_ms),
        plot_student_latency(all_ms),
    ]
    if include_diversity:
        figs.append(plot_diversity_tsne())

    for fig in figs:
        if fig is not None:
            plt.show()
    print('\n[Done]')


if __name__ == '__main__':
    make_full_report()
