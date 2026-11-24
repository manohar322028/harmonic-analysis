"""Poster versions of the climatology figures (CO_AGU2026 poster).

Same data and methods as mean_seasonal_cycle.ipynb, seasonal-cycle-eofs.ipynb and
harmonic.ipynb; figures are re-laid out at their exact poster size (inches) with poster-size
fonts and neutral titles. Run from Scripts/climatology with the gis-env interpreter.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from xeofs.single import EOF

OUT = "../../Figures/climatology/poster"
os.makedirs(OUT, exist_ok=True)
DPI = 220
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.titlesize": 15,
                     "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})

cycle = xr.open_dataarray("mean_seasonal_cycle.nc").load()
complete = xr.open_dataarray("complete_data.nc").load()
mnames = list(cycle["month_name"].values)
mpos = np.arange(1, 13)
lons, lats = cycle.lon.values, cycle.lat.values
EXT = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
BOXES = {
    1: dict(LAT0=31, LAT1=36, LON0=-117, LON1=-112),
    2: dict(LAT0=45, LAT1=49, LON0=-85,  LON1=-70),
    3: dict(LAT0=17, LAT1=19, LON0=-102, LON1=-99),
}
BLUE = "#1f77b4"


def box_mean(da, k):
    b = BOXES[k]
    return da.sel(lat=slice(b["LAT0"], b["LAT1"]), lon=slice(b["LON0"], b["LON1"])).mean(("lat", "lon"))


def base_map(ax, left=True, bottom=True, fs=11):
    ax.set_extent(EXT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="k")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#555")
    ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="#888")
    gl = ax.gridlines(draw_labels=True, linewidth=0, color="none", xlocs=[-120, -110, -100, -90, -80, -70],
                      ylocs=[15, 25, 35, 45])
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = left, bottom
    gl.xlabel_style = gl.ylabel_style = {"size": fs}


def draw_box(ax, k, color="#ccff00"):
    b = BOXES[k]
    ax.plot([b["LON0"], b["LON1"], b["LON1"], b["LON0"], b["LON0"]],
            [b["LAT0"], b["LAT0"], b["LAT1"], b["LAT1"], b["LAT0"]],
            "-", color=color, lw=2.2, transform=ccrs.PlateCarree())


def month_axis(ax, zero=True):
    ax.set_xticks(mpos); ax.set_xticklabels(mnames)
    if zero:
        ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.4)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI)          # exact figure size, no bbox cropping
    plt.close(fig)
    print("Saved ->", path, f"({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in)")


# --- Fig. 2: twelve monthly mean maps ---------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(16.6, 9.0), subplot_kw={"projection": ccrs.PlateCarree()},
                         layout="constrained")
vmin, vmax = np.nanpercentile(cycle.values, [2, 98])
for m, ax in enumerate(axes.flat, start=1):
    base_map(ax, left=(m - 1) % 4 == 0, bottom=m > 8)
    pcm = ax.pcolormesh(lons, lats, cycle.sel(month=m).values, transform=ccrs.PlateCarree(),
                        cmap="YlOrRd", vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_title(mnames[m - 1], fontsize=16, fontweight="bold", pad=4)
cb = fig.colorbar(pcm, ax=axes, orientation="vertical", shrink=0.75, pad=0.01, extend="both")
cb.set_label("CO (kg kg$^{-1}$)", fontsize=14)
save(fig, "monthly_mean_maps.png")

# --- Fig. 3: domain-mean seasonal cycle (±1 SD across years) ---------------------------------
ts = complete.mean(("lat", "lon")).to_series()
clim, sd = ts.groupby(ts.index.month).mean(), ts.groupby(ts.index.month).std()
fig, ax = plt.subplots(figsize=(11.2, 4.38), layout="constrained")
ax.fill_between(mpos, clim - sd, clim + sd, color=BLUE, alpha=0.2, label="±1 SD across years")
ax.plot(mpos, clim.values, "o-", lw=2.4, ms=7, color=BLUE, label="Domain mean")
month_axis(ax, zero=False)
ax.set_ylabel("CO (kg kg$^{-1}$)")
ax.legend(loc="upper right", fontsize=13)
save(fig, "domain_mean_seasonal_cycle.png")

# --- Fig. 4: EOF gain maps (top) and mode reconstructions over each box (bottom) -------------
model = EOF(n_modes=12, use_coslat=False)
model.fit(cycle, dim="month")
eofs, pcs = model.components(), model.scores()
expvar = model.explained_variance_ratio().values * 100

fig = plt.figure(figsize=(16.6, 8.3), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95])
for j, k in enumerate((1, 2, 3)):
    recon_k = box_mean(eofs.sel(mode=k) * pcs.sel(mode=k), k)                         # (month,)
    gain = (100 * xr.cov(cycle, recon_k, dim="month") / xr.cov(recon_k, recon_k, dim="month")).values
    ax = fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree())
    base_map(ax, left=j == 0, fs=10)
    neg = np.nanpercentile(gain, 2) < 0
    pcm = ax.pcolormesh(lons, lats, gain, transform=ccrs.PlateCarree(), shading="nearest",
                        cmap="RdBu_r" if neg else "Reds", vmin=-100 if neg else 0, vmax=100)
    draw_box(ax, k)
    ax.set_title(f"EOF {k}  ({expvar[k-1]:.1f}% var)", fontsize=15)
    cb = fig.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.78, pad=0.02)
    cb.set_label(f"Gain relative to box {k} (%)", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    axr = fig.add_subplot(gs[1, j])
    axr.plot(mpos, recon_k.values, "o-", lw=2, ms=5, color=BLUE)
    month_axis(axr)
    axr.set_xticklabels([n[0] for n in mnames])
    axr.set_title(f"Mode {k} reconstruction over box {k}", fontsize=14)
    if j == 0:
        axr.set_ylabel("CO (kg kg$^{-1}$)")
save(fig, "eof_gain_recon.png")
print("EOF variance (%):", np.round(expvar[:3], 1), " sum:", round(float(expvar[:3].sum()), 1))

# --- Fig. 5: 12-month harmonic amplitude ratio and relative phase vs box 1 -------------------
h12 = xr.open_dataset("harmonic_12m.nc").load()
z = h12["a_cos"] + 1j * h12["b_sin"]                     # a + ib: |z| = A, angle = phi
ref = box_mean(z, 1)
amp_ratio = 100 * np.abs(z.values) / np.abs(ref.values)
rel_phase = np.rad2deg(np.angle(z.values * np.conj(ref.values)))      # deg, > 0 = peaks later

relphase_colors = ["#4b72b0", "#e2847a", "#55a868", "#dd8452", "#8172b3", "#ccb974",
                   "#64b5cd", "#59a14f", "#b07aa1", "#8e44ad", "#c49a6c", "#ffee33"]
pcmap = mcolors.LinearSegmentedColormap.from_list("relphase", relphase_colors + relphase_colors[:1], N=360)
dcmap = mcolors.ListedColormap([pcmap((v % 360) / 360) for v in np.linspace(-360, 360, 721)])
dnorm = mcolors.Normalize(vmin=-360, vmax=360)

fig, axes = plt.subplots(1, 2, figsize=(16.6, 4.75), subplot_kw={"projection": ccrs.PlateCarree()},
                         layout="constrained")
base_map(axes[0])
pcm = axes[0].pcolormesh(lons, lats, amp_ratio, transform=ccrs.PlateCarree(), cmap="Reds",
                         vmin=0, vmax=100, shading="nearest")
cb = fig.colorbar(pcm, ax=axes[0], shrink=0.85, pad=0.02)
cb.set_label("Amplitude relative to box 1 (%)")
axes[0].set_title("12-month harmonic relative amplitude")

base_map(axes[1], left=False)
pcm = axes[1].pcolormesh(lons, lats, rel_phase, transform=ccrs.PlateCarree(), cmap=dcmap, norm=dnorm,
                         shading="nearest")
cb = fig.colorbar(pcm, ax=axes[1], shrink=0.85, pad=0.02)
lo, hi = np.floor(np.nanmin(rel_phase) / 30) * 30, np.ceil(np.nanmax(rel_phase) / 30) * 30
cb.ax.set_ylim(lo, hi)
ticks = np.arange(lo, hi + 1e-6, 30)
cb.set_ticks(ticks)
cb.set_ticklabels([f"{int(t)}°" for t in ticks])
cb.set_label("Phase relative to box 1\n(30° = 1 month, + = later)")
axes[1].set_title("12-month harmonic relative phase")
for ax in axes:
    draw_box(ax, 1)
save(fig, "harmonic12_relamp_relphase_box1.png")

# --- Fig. 6: 12-month harmonic reconstruction over box 1 -------------------------------------
recon12 = box_mean(xr.open_dataset("periodicities.nc")["harmonic_12m"], 1)
fig, ax = plt.subplots(figsize=(10.0, 3.8), layout="constrained")
ax.plot(mpos, recon12.values, "o-", lw=2.2, ms=6, color=BLUE)
month_axis(ax)
ax.set_ylabel("CO (kg kg$^{-1}$)")
ax.set_title("12-month harmonic reconstruction over box 1", fontsize=15)
save(fig, "harmonic12_recon_box1.png")

pv = xr.open_dataset("periodicities.nc")
print(f"12-month harmonic share of seasonal variance: {pv.attrs['variance_explained_12m_percent']:.1f}%")
pk = cycle.argmax("month") + 1
may = pk.where(pk == 5, drop=True)
print("May-peaking grid points:", int((pk == 5).sum()), "| lat range", float(may.lat.min()), "-", float(may.lat.max()),
      "| share south of 21N:", f"{100*float((pk == 5).sel(lat=slice(0, 21)).sum() / (pk == 5).sum()):.0f}%")
