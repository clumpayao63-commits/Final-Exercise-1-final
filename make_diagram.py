import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

fig, ax = plt.subplots(figsize=(11, 7)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 7)
ax.add_patch(Rectangle((3, .3), 5, 6.4, fill=False, lw=2))
ax.text(5.5, 6.4, "SLA Breach Monitoring System", ha="center", weight="bold")
cases = ["Upload & validate CSV", "Explore data (EDA)", "Predict SLA breach", "View risk dashboard", "Record review decision", "View system design / audit"]
ys = [5.6, 4.7, 3.8, 2.9, 2.0, 1.1]
for t, y in zip(cases, ys):
    ax.add_patch(Ellipse((5.5, y), 3.6, .7, fill=False)); ax.text(5.5, y, t, ha="center", va="center", fontsize=9)
actors = {"Student Services\nStaff": (1.2, 3.8, [0, 1, 2, 3, 4]), "Supervisor": (1.2, 1.4, [3, 4]),
          "System\nAdministrator": (9.8, 3.0, [0, 5])}
for name, (x, y, links) in actors.items():
    ax.plot([x], [y + .35], "o", ms=14, mfc="white", mec="k"); ax.plot([x, x], [y + .2, y - .4], "k"); ax.text(x, y - .8, name, ha="center", fontsize=9)
    for i in links:
        ax.plot([x + (.2 if x < 5 else -.2), 5.5 + (-1.8 if x < 5 else 1.8)], [y, ys[i]], "k-", lw=.6)
ax.text(5.5, .55, "<<include>> Predict uses saved model pipeline; Record decision requires human review", ha="center", fontsize=8, style="italic")
plt.savefig("assets/use_case_diagram.png", dpi=150, bbox_inches="tight")
