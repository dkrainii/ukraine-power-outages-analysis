"""Figures and numbers for the paper. Run from the paper/ folder after notebook 01 has created
data/clean/hourly_dataset.csv. Reproduces the analysis of notebooks 01-02 and writes numbers.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SEED = 42
N_BOOT = 300                      # block-bootstrap replications for the intervals of odds ratios
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
PNG = Path("../docs/img"); PNG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL, FULL = 3.35, 7.0             # column and page width, inches
pct = mticker.PercentFormatter(decimals=0)


def savefig(fig, name):
    """PDF for the paper, PNG for the README."""
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PNG / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


N = {}                            # every number quoted in the paper

raw = pd.read_csv("../data/clean/hourly_dataset.csv", parse_dates=["dt"])
N["hours_total"] = len(raw)
N["outage_hours_total"] = int(raw["outage"].sum())
N["agg_hours"] = int(raw["aggregated_period"].sum())
df = raw[raw["aggregated_period"] == 0].reset_index(drop=True)
df["date"] = df["dt"].dt.normalize()
N["hours_used"] = len(df); N["outage_share"] = df["outage"].mean()

# ---------- Fig 1: timeline ----------
strikes = pd.read_csv("../data/raw/2026_07_01_udary_krytychna_infrastruktura.csv", parse_dates=["date"])
att = strikes.groupby("date")["weapon_fired"].sum()
N["attacks_total"] = int(len(att)); N["attacks_window"] = int((att.index >= "2023-01-01").sum())
N["attack_first"] = str(att.index.min().date())
att = att[att.index >= "2023-01-01"]
weekly = raw.set_index("dt")["outage"].resample("W").sum()
agg_w = raw.set_index("dt")["aggregated_period"].resample("W").max()
N["monthly_outage_hours"] = {str(k): int(v) for k, v in raw.groupby(raw["dt"].dt.to_period("M"))["outage"].sum().items() if v > 0}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(FULL, 2.9), sharex=True, gridspec_kw={"hspace": 0.25})
a1.vlines(att.index, 0, att.values, color=BLUE, lw=1.2); a1.scatter(att.index, att.values, s=5, color=BLUE, zorder=3)
a1.set_ylabel("missiles + drones\nper attack"); a1.set_ylim(0); a1.grid(axis="x", visible=False)
a1.set_title("(a) Massive attacks on energy infrastructure", loc="left")
a2.bar(weekly.index, weekly.values, width=6, color=np.where(agg_w.values == 1, "#c3c2b7", ORANGE))
a2.set_ylabel("outage hours\nper week"); a2.set_ylim(0, 172); a2.grid(axis="x", visible=False)
a2.set_title("(b) Hours with stabilisation outages; grey = aggregated reporting period", loc="left")
savefig(fig, "fig1_timeline")

# ---------- Fig 2: heatmap ----------
heat = (df.assign(ym=df["dt"].dt.to_period("M").astype(str))
        .pivot_table(index="hour", columns="ym", values="outage", aggfunc="mean") * 100)
fig, ax = plt.subplots(figsize=(COL, 2.1))
im = ax.imshow(heat.values, aspect="auto", cmap=SEQ, vmin=0, vmax=100, origin="lower")
ax.set_xticks(range(0, heat.shape[1], 6)); ax.set_xticklabels(heat.columns[::6], rotation=45, ha="right")
ax.set_yticks(range(0, 24, 4)); ax.set_ylabel("hour of day"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("% of days with outage"); cb.outline.set_visible(False)
savefig(fig, "fig2_heatmap")

# ---------- Fig 8: daily profile, all days vs days with partial outages ----------
per_day = df.groupby("date")["outage"].agg(["sum", "size"])
full_days = per_day.index[per_day["sum"] == per_day["size"]]
part_days = per_day.index[(per_day["sum"] > 0) & (per_day["sum"] < per_day["size"])]
prof_all = df.groupby("hour")["outage"].mean() * 100
prof_part = df[df["date"].isin(part_days)].groupby("hour")["outage"].mean() * 100
N["days"] = len(per_day); N["days_any"] = int((per_day["sum"] > 0).sum())
N["days_full"] = len(full_days); N["days_partial"] = len(part_days)
N["full_days_share_of_outage_hours"] = per_day.loc[full_days, "sum"].sum() / per_day["sum"].sum()
N["hour_profile"] = prof_all.round(1).to_dict(); N["hour_profile_partial"] = prof_part.round(1).to_dict()
fig, ax = plt.subplots(figsize=(COL, 1.9))
ax.plot(prof_all.index, prof_all.values, color=BLUE, lw=1.4, label="all days")
ax.plot(prof_part.index, prof_part.values, color=ORANGE, lw=1.4, label=f"days with partial outages (n = {len(part_days)})")
ax.set_xlabel("hour of day"); ax.set_ylabel("% hours with outage"); ax.set_xticks(range(0, 24, 3)); ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(pct); ax.legend(frameon=False, loc="upper left")
savefig(fig, "fig8_hour_profile")

# ---------- Fig 3: attack history ----------
lag = (df.assign(b=pd.cut(df["days_since_attack"], [-1, 2, 7, 14, 30, 60, 10000],
                         labels=["0-2", "3-7", "8-14", "15-30", "31-60", ">60"]))
       .groupby("b", observed=True)["outage"].mean() * 100)
cnt = (df.assign(b=pd.cut(df["attacks_30d"], [-1, 0, 1, 2, 4, 100], labels=["0", "1", "2", "3-4", "5+"]))
       .groupby("b", observed=True)["outage"].mean() * 100)
N["lag"] = lag.round(1).to_dict(); N["cnt"] = cnt.round(1).to_dict()
late = df[(df["attacks_30d"] == 0) & (df["outage"] == 1)]
N["late_outage_hours"] = len(late)
N["late_outage_days"] = {str(k.date()): int(v) for k, v in late.groupby("date")["days_since_attack"].first().items()}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.8), sharey=True)
a1.bar(range(len(lag)), lag.values, color=BLUE, width=0.7); a1.set_xticks(range(len(lag))); a1.set_xticklabels(lag.index, rotation=45, ha="right")
a1.set_xlabel("days since last attack"); a1.set_ylabel("% hours with outage")
a2.bar(range(len(cnt)), cnt.values, color=BLUE, width=0.7); a2.set_xticks(range(len(cnt))); a2.set_xticklabels(cnt.index)
a2.set_xlabel("attacks in previous 30 days")
for a in (a1, a2): a.grid(axis="x", visible=False); a.yaxis.set_major_formatter(pct)
savefig(fig, "fig3_attack_history")

# ---------- Fig 4: temperature x attacks ----------
tg = pd.cut(df["temp_c"], [-40, -5, 5, 15, 25, 50], labels=["< -5", "-5..5", "5..15", "15..25", "> 25"])
ag = np.where(df["attacks_30d"] > 0, "attacked", "none")
cross = df.assign(t=tg, a=ag).pivot_table(index="t", columns="a", values="outage", aggfunc=["mean", "size"], observed=True)
share = cross["mean"] * 100; hours = cross["size"]
N["cross_share"] = share.round(1).to_dict(); N["cross_hours"] = hours.to_dict()
fig, ax = plt.subplots(figsize=(COL, 2.0))
xs = np.arange(len(share)); w = 0.38
ax.bar(xs - w / 2, share["none"], w, color="#86b6ef", label="no attack in previous 30 days")
ax.bar(xs + w / 2, share["attacked"], w, color=BLUE, label="at least one attack in previous 30 days")
for off, c in [(-w / 2, "none"), (w / 2, "attacked")]:
    for xi, v in zip(xs, share[c]): ax.text(xi + off, v + 1.2, f"{v:.0f}", ha="center", fontsize=6, color=INK2)
ax.set_xticks(xs); ax.set_xticklabels(share.index); ax.set_xlabel("air temperature, °C")
ax.set_ylabel("% hours with outage"); ax.set_ylim(0, 75); ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(pct); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
savefig(fig, "fig4_temp_attacks")

# U-shape overall
t = df.assign(b=pd.cut(df["temp_c"], list(range(-25, 41, 5)))).groupby("b", observed=True)["outage"].agg(["mean", "size"])
N["temp_bins"] = {str(k): round(v * 100, 1) for k, v in t[t["size"] >= 50]["mean"].items()}

# ---------- Daily tests ----------
daily = df.groupby("date").agg(share=("outage", "mean"), any_outage=("outage", "max"),
    temp=("temp_c", "mean"), attacks_30d=("attacks_30d", "first"), days_since=("days_since_attack", "first"),
    weekend=("is_weekend", "first"))
daily["attacked"] = (daily["attacks_30d"] > 0).astype(int)
ct = pd.crosstab(daily["attacked"], daily["any_outage"])
chi2, p, _, _ = stats.chi2_contingency(ct)
N["ct"] = ct.values.tolist(); N["chi2"] = chi2; N["chi2_p"] = p; N["V"] = float(np.sqrt(chi2 / ct.values.sum()))
N["spearman"] = stats.spearmanr(daily["days_since"], daily["share"])[:2]
near = daily[daily["days_since"] <= 60]; N["spearman60"] = stats.spearmanr(near["days_since"], near["share"])[:2]
a = daily[daily["attacked"] == 1]
cold, mild, hot = a.loc[a["temp"] < 0, "share"], a.loc[a["temp"].between(5, 20), "share"], a.loc[a["temp"] > 22, "share"]
N["cold"] = (len(cold), cold.mean(), stats.mannwhitneyu(cold, mild, alternative="greater").pvalue)
N["mild"] = (len(mild), mild.mean())
N["hot"] = (len(hot), hot.mean(), stats.mannwhitneyu(hot, mild, alternative="greater").pvalue)
wk, wd = a.loc[a["weekend"] == 1, "share"], a.loc[a["weekend"] == 0, "share"]
N["weekend"] = (wd.mean(), wk.mean(), stats.mannwhitneyu(wd, wk, alternative="greater").pvalue, len(wd), len(wk))

# the same weather in different seasons: cold and hot days after attacks, by quarter
a = a.assign(q=pd.PeriodIndex(a.index, freq="Q"))
by_q = lambda d: {str(k): [int(r["n"]), round(r["share"] * 100, 1), round(r["temp"], 1)] for k, r in
                  d.groupby("q").agg(n=("share", "size"), share=("share", "mean"), temp=("temp", "mean")).iterrows()}
N["cold_days_by_quarter"] = by_q(a[a["temp"] < 0]); N["hot_days_by_quarter"] = by_q(a[a["temp"] > 22])

q = df.groupby(df["dt"].dt.to_period("Q")).agg(attacks=("attack_day", lambda s: int(s.sum() / 24)),
    fired30=("fired_30d", "mean"), outage=("outage", "mean"), temp=("temp_c", "mean"))
N["quarters"] = {str(k): [int(r.attacks), round(r.fired30), round(r.outage * 100, 1), round(r.temp, 1)] for k, r in q.iterrows()}
N["attack_day_hours"] = int(df["attack_day"].sum())

# ---------- Models ----------
def add(d):
    """Derived features of the logistic regression, computed from the base columns."""
    d = d.copy()
    d["cold_deg"] = (10 - d["temp_c"]).clip(lower=0); d["heat_deg"] = (d["temp_c"] - 20).clip(lower=0)
    d["log_days_since"] = np.log1p(d["days_since_attack"]); d["recent_attacks"] = (d["attacks_30d"] > 0).astype(int)
    d["cold_x_recent"] = d["cold_deg"] * d["recent_attacks"]; d["heat_x_recent"] = d["heat_deg"] * d["recent_attacks"]
    d["sun_x_recent"] = d["radiation_wm2"] * d["recent_attacks"]
    return d

df = add(df)
NUM = ["cold_deg", "heat_deg", "radiation_wm2", "cloud_pct", "wind_kmh", "is_weekend", "log_days_since",
       "attacks_30d", "recent_attacks", "cold_x_recent", "heat_x_recent", "sun_x_recent"]
ATT = ["attacks_30d", "log_days_since", "recent_attacks"]            # baseline: attack history only
XG = ["hour", "dow", "temp_c", "radiation_wm2", "cloud_pct", "wind_kmh", "days_since_attack", "attacks_7d",
      "attacks_30d", "fired_30d", "not_down_30d"]
XG_NOW = [c for c in XG if c not in ("fired_30d", "not_down_30d")]   # without the numbers of weapons


def mk_lr(num=NUM, hour=True):
    parts = [("n", StandardScaler(), num)] + ([("h", OneHotEncoder(drop="first"), ["hour"])] if hour else [])
    return Pipeline([("p", ColumnTransformer(parts)), ("m", LogisticRegression(max_iter=3000))])


def mk_gb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=SEED)


MODELS = {"Attack history only": (lambda: mk_lr(ATT, hour=False), ATT),
          "Logistic regression": (mk_lr, NUM + ["hour"]),
          "Gradient boosting": (mk_gb, XG),
          "Gradient boosting, no weapon counts": (mk_gb, XG_NOW)}
XL = NUM + ["hour"]


def evaluate(train, test):
    """Fit every model on train, return predictions for test."""
    P = {"Base rate": np.full(len(test), train["outage"].mean())}
    for name, (make, cols) in MODELS.items():
        P[name] = make().fit(train[cols], train["outage"]).predict_proba(test[cols])[:, 1]
    return P


def scores(y, p):
    return [roc_auc_score(y, p) if p.std() > 0 else 0.5, average_precision_score(y, p), brier_score_loss(y, p), p.mean()]


tr, te = df[df["dt"] < "2025-10-01"], df[df["dt"] >= "2025-10-01"]
N["train"] = (len(tr), tr["outage"].mean()); N["test"] = (len(te), te["outage"].mean())
P = evaluate(tr, te)
N["metrics"] = {k: scores(te["outage"], p) for k, p in P.items()}
lr = mk_lr().fit(tr[XL], tr["outage"])

# calibration of the logistic regression on the test season
cal = te.assign(p=P["Logistic regression"])
m = cal.groupby(cal["dt"].dt.to_period("M")).agg(hours=("outage", "size"), obs=("outage", "mean"), pred=("p", "mean"))
N["calibration_by_month"] = {str(k): [int(r.hours), round(r.obs * 100, 1), round(r.pred * 100, 1)] for k, r in m.iterrows()}

# leave-one-wave-out; the fourth wave coincides with the test season
waves = pd.cut(df["dt"], pd.to_datetime(["2023-01-01", "2023-10-01", "2024-10-01", "2025-10-01", "2026-07-01"]),
               right=False, labels=["W1", "W2", "W3", "W4"])
cv = {}
for w in waves.cat.categories:
    a_, b_ = df[waves != w], df[waves == w]
    Pw = evaluate(a_, b_); y = b_["outage"]
    cv[w] = {"hours": len(b_), "share": y.mean(),
             "auc": {k: roc_auc_score(y, p) for k, p in Pw.items() if k != "Base rate"},
             "mean_p_lr": Pw["Logistic regression"].mean(),
             "brier_lr": brier_score_loss(y, Pw["Logistic regression"]),
             "brier_base": brier_score_loss(y, Pw["Base rate"])}
N["cv"] = cv
N["cv_mean_auc"] = {k: float(np.mean([cv[w]["auc"][k] for w in cv])) for k in MODELS}

# permutation importance of feature groups (logistic regression, test season)
rng = np.random.default_rng(SEED); base = te.reset_index(drop=True)
b_auc = roc_auc_score(base["outage"], lr.predict_proba(base[XL])[:, 1]); imp = {}
for g, cols in {"attack history": ["days_since_attack", "attacks_30d"], "temperature": ["temp_c"], "hour of day": ["hour"],
                "weekend": ["is_weekend"], "solar radiation and cloud": ["radiation_wm2", "cloud_pct"], "wind": ["wind_kmh"]}.items():
    d_ = []
    for _ in range(5):
        Xp = base.copy(); Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]; Xp = add(Xp)
        d_.append(b_auc - roc_auc_score(base["outage"], lr.predict_proba(Xp[XL])[:, 1]))
    imp[g] = float(np.mean(d_))
N["importance"] = imp

# ---------- Interpretation: odds ratios in natural units ----------
# Radiation as a deviation from its usual value for the month and hour: a robustness check for the solar effect
df["rad_anom"] = df["radiation_wm2"] - df.groupby([df["dt"].dt.month, "hour"])["radiation_wm2"].transform("mean")
df["anom_x_recent"] = df["rad_anom"] * df["recent_attacks"]
NUM_ANOM = [c for c in NUM if c not in ("radiation_wm2", "sun_x_recent", "cloud_pct")] + ["rad_anom", "anom_x_recent"]


def raw_betas(model, num):
    """Coefficients per one natural unit of each numeric feature."""
    scale = model.named_steps["p"].named_transformers_["n"].scale_
    return dict(zip(num, model.named_steps["m"].coef_[0][:len(num)] / scale))


def effects(d):
    """Odds ratios for interpretable changes. Weather effects are given for hours after recent attacks,
    i.e. as the sum of the main coefficient and the interaction."""
    b = raw_betas(mk_lr().fit(d[XL], d["outage"]), NUM)
    ba = raw_betas(mk_lr(NUM_ANOM).fit(d[NUM_ANOM + ["hour"]], d["outage"]), NUM_ANOM)
    return {
        "10 °C colder, below 10 °C": np.exp(10 * (b["cold_deg"] + b["cold_x_recent"])),
        "5 °C hotter, above 20 °C": np.exp(5 * (b["heat_deg"] + b["heat_x_recent"])),
        "one more attack in 30 days": np.exp(b["attacks_30d"]),
        "+100 W/m² solar radiation": np.exp(100 * (b["radiation_wm2"] + b["sun_x_recent"])),
        "twice as many days since attack": np.exp(np.log(2) * b["log_days_since"]),
        "weekend": np.exp(b["is_weekend"]),
        "+50 points of cloud cover": np.exp(50 * b["cloud_pct"]),
        "+10 km/h wind": np.exp(10 * b["wind_kmh"]),
        "solar_anomaly_100": np.exp(100 * (ba["rad_anom"] + ba["anom_x_recent"])),
    }


lrf = mk_lr().fit(df[XL], df["outage"])
per_sd = pd.Series(np.exp(lrf.named_steps["m"].coef_[0][:len(NUM)]), index=NUM)
N["odds_per_sd"] = per_sd.round(2).to_dict()
N["no_attack_cold_hours"] = (int(len(df[(df["attacks_30d"] == 0) & (df["cold_deg"] > 0)])),
                             int(df.loc[(df["attacks_30d"] == 0) & (df["cold_deg"] > 0), "outage"].sum()))
point = effects(df)

# Block bootstrap: calendar months are resampled with replacement, because outage periods last for weeks
months = df["dt"].dt.to_period("M"); blocks = {k: g for k, g in df.groupby(months)}; keys = list(blocks)
rng = np.random.default_rng(SEED); boot = []
for _ in range(N_BOOT):
    sample = pd.concat([blocks[keys[i]] for i in rng.choice(len(keys), len(keys))], ignore_index=True)
    boot.append(effects(sample))
boot = pd.DataFrame(boot)
ci = boot.quantile([0.025, 0.975])
N["effects"] = {k: [round(float(point[k]), 2), round(float(ci.loc[0.025, k]), 2), round(float(ci.loc[0.975, k]), 2)] for k in point}

shown = [k for k in point if k != "solar_anomaly_100"]
shown = sorted(shown, key=lambda k: point[k])
fig, ax = plt.subplots(figsize=(COL, 2.2))
ys = np.arange(len(shown)); v = np.array([point[k] for k in shown])
lo = np.array([ci.loc[0.025, k] for k in shown]); hi = np.array([ci.loc[0.975, k] for k in shown])
ax.hlines(ys, lo, hi, color="#86b6ef", lw=2.2); ax.scatter(v, ys, s=14, color=BLUE, zorder=3)
ax.axvline(1, color=MUTED, lw=0.8); ax.set_xscale("log"); ax.set_yticks(ys); ax.set_yticklabels(shown)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}")); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("odds ratio (log scale)"); ax.grid(axis="y", visible=False)
savefig(fig, "fig5_odds")

# ---------- Partial dependence (full-data logistic regression) ----------
ref = df[df["attacks_30d"] > 0].sample(4000, random_state=SEED)


def pdp(feat, grid):
    out = []
    for val in grid:
        X = ref.copy(); X[feat] = val; out.append(lrf.predict_proba(add(X)[XL])[:, 1].mean())
    return np.array(out) * 100


specs = [("temp_c", np.arange(-15, 34), "temperature, °C"), ("radiation_wm2", np.arange(0, 851, 25), "solar radiation, W/m²")]
fig, axes = plt.subplots(1, 2, figsize=(COL, 1.7), sharey=True)
pdv = {}
for ax, (f, g, lab) in zip(axes, specs):
    val = pdp(f, g); pdv[f] = (g.tolist(), val.round(1).tolist())
    ax.plot(g, val, color=BLUE, lw=1.4); ax.set_xlabel(lab)
axes[0].set_xticks([-10, 0, 10, 20, 30]); axes[0].yaxis.set_major_formatter(pct)
axes[0].set_ylabel("P(outage)"); axes[0].set_ylim(0, 100)
savefig(fig, "fig6_pdp")
N["pdp_temp_min"] = float(min(pdv["temp_c"][1])); N["pdp_temp_ends"] = (pdv["temp_c"][1][0], pdv["temp_c"][1][-1])
N["pdp_rad"] = (pdv["radiation_wm2"][1][0], pdv["radiation_wm2"][1][-1])

# ---------- Scenarios ----------
month = df["dt"].dt.month
rad = df.groupby([month, "hour"])["radiation_wm2"].mean(); cld = df.groupby([month, "hour"])["cloud_pct"].mean()
anom = (df["temp_c"] - df.groupby("date")["temp_c"].transform("mean")).groupby([month, df["hour"]]).mean()


def scen(m, t, ds, a30):
    s = pd.DataFrame({"hour": range(24)}); s["is_weekend"] = 0; s["temp_c"] = t + anom.loc[m].values
    s["radiation_wm2"] = rad.loc[m].values; s["cloud_pct"] = cld.loc[m].values; s["wind_kmh"] = df["wind_kmh"].median()
    s["days_since_attack"] = ds; s["attacks_30d"] = a30
    return lrf.predict_proba(add(s)[XL])[:, 1] * 100


S = {"January, mean −8 °C, 2 days after attack": scen(1, -8, 2, 3), "July, mean +24 °C, 2 days after attack": scen(7, 24, 2, 3),
     "April, mean +12 °C, 2 days after attack": scen(4, 12, 2, 3), "January, mean −8 °C, no attack for 90 days": scen(1, -8, 90, 0)}
N["scen"] = {k: [round(float(v.min()), 1), int(v.argmin()), round(float(v.max()), 1), int(v.argmax())] for k, v in S.items()}
N["scen_hours"] = {k: v.round(0).tolist() for k, v in S.items()}
fig, ax = plt.subplots(figsize=(COL, 2.0))
for (k, v), c in zip(S.items(), [BLUE, ORANGE, AQUA, YELLOW]): ax.plot(range(24), v, color=c, lw=1.4, label=k)
ax.set_xlabel("hour of day"); ax.set_ylabel("P(outage)"); ax.set_xticks(range(0, 24, 3)); ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(pct)
ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.28), handlelength=1.6)
savefig(fig, "fig7_scenarios")

dump = lambda o: o.item() if hasattr(o, "item") else str(o)
json.dump(N, open("numbers.json", "w", encoding="utf-8"), indent=1, default=dump, ensure_ascii=False)
print(json.dumps(N, indent=1, default=dump, ensure_ascii=False))
