from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
