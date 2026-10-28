"""
fig_well_forgetting_dual, reduced to the ABSOLUTE-change panel only.

Nothing else changed: same data, colours, phase shading, dividers, tick
placement and right-side value labels as the right-hand panel of the dual
figure. Data loading is imported from forgetting_figures.py (which has a
__main__ guard) so there is no second copy of the loader to drift.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forgetting_figures import well_data, DS, OUT, P1_COLOR, P2_COLOR, C

PHASE_DIV = 51

all_eps = set()
for ds in DS:
    all_eps.update(well_data[ds["name"]].keys())
xmax = max(all_eps)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axvspan(0, PHASE_DIV, alpha=0.06, color=P1_COLOR, lw=0)
ax.axvspan(PHASE_DIV, xmax, alpha=0.06, color=P2_COLOR, lw=0)
ax.axvline(PHASE_DIV, color=C["phase_div"], lw=1.0, ls="--", alpha=0.7)

ends = []
for ds in DS:
    ep_nrmse = well_data[ds["name"]]
    if not ep_nrmse:
        continue
    pretrained = ep_nrmse.get(0)
    if pretrained is None:
        continue
    eps = sorted(ep_nrmse)
    delta = [ep_nrmse[e] - pretrained for e in eps]
    ax.plot(eps, delta, color=ds["color"], lw=2, marker="o", ms=5, zorder=3,
            label=ds["name"])
    ends.append([delta[-1], eps[-1], ds])

# The single-panel aspect brings Turbcool and MHD-64 end labels within a few
# thousandths of each other; nudge overlapping labels apart in y so neither is
# unreadable. Values and line positions are untouched.
ends.sort(key=lambda e: e[0])
span = max(e[0] for e in ends) - min(e[0] for e in ends)
minsep = 0.055 * span
for i in range(1, len(ends)):
    if ends[i][0] - ends[i - 1][0] < minsep:
        ends[i][0] = ends[i - 1][0] + minsep
for ylab, xend, ds in ends:
    ytrue = well_data[ds["name"]][xend] - well_data[ds["name"]][0]
    ax.annotate(f"{ds['name']}  Δ{ytrue:.4f}",
                xy=(xend, ylab), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=ds["color"], va="center", ha="left",
                annotation_clip=False)

xform = ax.get_xaxis_transform()
ax.text(PHASE_DIV / 2, 0.02, "Phase 1", ha="center", va="bottom", fontsize=9,
        color=P1_COLOR, style="italic", transform=xform)
ax.text((PHASE_DIV + xmax) / 2, 0.02, "Phase 2", ha="center", va="bottom",
        fontsize=9, color=P2_COLOR, style="italic", transform=xform)

ax.set_xlabel("Total fine-tuning epochs from pretrain")
ax.set_ylabel("NRMSE absolute change (vs. PT)")
ax.set_xlim(left=-3, right=xmax + 2)
ax.set_ylim(bottom=0)
tick_eps = sorted(all_eps)
ax.set_xticks(tick_eps)
ax.set_xticklabels([str(e) for e in tick_eps], rotation=45, ha="right", fontsize=9)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_well_forgetting_abs.{ext}"), bbox_inches="tight")
plt.close(fig)
print(f"\nSaved fig_well_forgetting_abs.{{pdf,png}} to {OUT}")
for ds in DS:
    d = well_data[ds["name"]]
    if d and 0 in d:
        print(f"  {ds['name']:15} P1(0->51) delta={d[51]-d[0]:.4f}   "
              f"P2(51->151) delta={d[151]-d[51]:.4f}")
