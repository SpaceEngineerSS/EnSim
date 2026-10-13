"""Visual tokens and the stylesheet built from them.

Two palettes share one structure: Console (dark, the default) and Paper
(light, for bright rooms and printed screenshots). Every colour in the
interface and in the plots comes from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"

SPACE = (0, 4, 8, 12, 16, 24, 32)
RADIUS = 2


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str  # window
    surface: str  # panels
    raised: str  # inputs, hovered rows
    line: str  # hairlines
    line_strong: str  # focused outlines, axis spines
    text: str
    muted: str  # secondary text, units
    faint: str  # disabled, gridlines
    signal: str  # the one accent: running state, primary action, key trace
    signal_text: str  # text on signal
    ok: str
    warn: str
    error: str
    series: tuple[str, ...]  # plot traces after the signal colour


CONSOLE = Palette(
    name="console",
    bg="#16181b",
    surface="#1c1f23",
    raised="#23272c",
    line="#2a2d31",
    line_strong="#3a3e44",
    text="#e6e3dd",
    muted="#8b8f94",
    faint="#55595f",
    signal="#e8743b",
    signal_text="#16181b",
    ok="#7fae6e",
    warn="#d9a441",
    error="#d0564a",
    series=("#e8743b", "#8fb3c9", "#c9c3b5", "#7fae6e", "#b48ead"),
)

PAPER = Palette(
    name="paper",
    bg="#f4f1ea",
    surface="#fbf9f4",
    raised="#ffffff",
    line="#ddd8cc",
    line_strong="#bdb6a6",
    text="#1f2124",
    muted="#6b6e72",
    faint="#a7a49c",
    signal="#c4561f",
    signal_text="#ffffff",
    ok="#4f7d3f",
    warn="#9a6d10",
    error="#b23a2e",
    series=("#c4561f", "#2f5d7c", "#5d5a52", "#4f7d3f", "#7a4f73"),
)

PALETTES = {CONSOLE.name: CONSOLE, PAPER.name: PAPER}

_current = CONSOLE


def current() -> Palette:
    return _current


def load_fonts() -> None:
    """Register the bundled IBM Plex faces with Qt."""
    from PyQt6.QtGui import QFontDatabase

    for path in sorted((ASSETS / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(path))


def stylesheet(p: Palette) -> str:
    """Application stylesheet for palette ``p``."""
    return f"""
* {{ outline: 0; }}
QWidget {{
    color: {p.text};
    font-family: "{SANS}";
    font-size: 13px;
    selection-background-color: {p.signal};
    selection-color: {p.signal_text};
}}
QMainWindow, QDialog, QStackedWidget, QSplashScreen, QMessageBox {{ background: {p.bg}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QMainWindow::separator {{ background: {p.line}; width: 1px; height: 1px; }}
QToolTip {{
    background: {p.raised};
    color: {p.text};
    border: 1px solid {p.line_strong};
    padding: 4px 6px;
}}

/* Structure */
#NavRail {{ background: {p.surface}; border-right: 1px solid {p.line}; }}
#Header {{ background: {p.bg}; border-bottom: 1px solid {p.line}; }}
#InputsPanel {{ background: {p.surface}; border-right: 1px solid {p.line}; }}
#InputsPanel QScrollArea, #InputsPanel QScrollArea > QWidget > QWidget {{ background: {p.surface}; }}
#StatusStrip {{ background: {p.bg}; border-top: 1px solid {p.line}; }}
#StatusStrip QLabel {{ color: {p.muted}; font-size: 12px; }}

QLabel#Title {{ font-size: 15px; font-weight: 600; }}
QLabel#SectionLabel {{
    color: {p.muted};
    font-size: 11px;
    font-weight: 600;
    padding-top: 12px;
}}
QLabel#Muted {{ color: {p.muted}; }}
QLabel#Mono {{ font-family: "{MONO}"; }}
QLabel#Hint {{ color: {p.muted}; font-size: 12px; }}
QLabel#Error {{ color: {p.error}; }}
QLabel#Warn {{ color: {p.warn}; }}
QLabel#Ok {{ color: {p.ok}; }}

/* Navigation rail */
#NavRail QToolButton {{
    background: transparent;
    border: none;
    border-left: 2px solid transparent;
    color: {p.muted};
    padding: 10px 4px 8px 2px;
    font-size: 11px;
}}
#NavRail QToolButton:hover {{ color: {p.text}; background: {p.raised}; }}
#NavRail QToolButton:checked {{ color: {p.text}; border-left: 2px solid {p.signal}; }}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p.raised};
    border: 1px solid {p.line};
    border-radius: {RADIUS}px;
    padding: 4px 6px;
    min-height: 20px;
}}
QSpinBox, QDoubleSpinBox {{ font-family: "{MONO}"; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {p.line_strong};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {p.faint};
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0; border: none; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: url({_icon_url("chevron-down", p.muted)}); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background: {p.raised};
    border: 1px solid {p.line_strong};
    selection-background-color: {p.line};
    selection-color: {p.text};
    padding: 2px;
}}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 12px; height: 12px;
    border: 1px solid {p.line_strong};
    background: {p.raised};
    border-radius: {RADIUS}px;
}}
QCheckBox::indicator:checked {{ background: {p.signal}; border-color: {p.signal}; }}

/* Buttons */
QPushButton, QToolButton {{
    background: {p.raised};
    border: 1px solid {p.line};
    border-radius: {RADIUS}px;
    padding: 5px 12px;
}}
QPushButton:hover, QToolButton:hover {{ border-color: {p.line_strong}; }}
QPushButton:pressed {{ background: {p.line}; }}
QPushButton:disabled {{ color: {p.faint}; }}
QPushButton#Primary {{
    background: {p.signal};
    color: {p.signal_text};
    border: 1px solid {p.signal};
    font-weight: 600;
    padding: 7px 14px;
}}
QPushButton#Primary:disabled {{ background: {p.line}; border-color: {p.line}; color: {p.muted}; }}
QPushButton#Quiet {{ background: transparent; border: none; color: {p.muted}; padding: 4px 6px; }}
QPushButton#Quiet:hover {{ color: {p.text}; }}

/* Tabs: underline, no boxes */
QTabWidget::pane {{ border: none; border-top: 1px solid {p.line}; top: -1px; }}
QTabWidget::tab-bar {{ left: 18px; }}
QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    color: {p.muted};
    padding: 8px 2px 7px 2px;
    margin-right: 18px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {p.text}; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.signal}; }}

/* Tables */
QTableWidget, QTableView, QTreeView, QListWidget {{
    background: {p.bg};
    alternate-background-color: {p.surface};
    border: 1px solid {p.line};
    gridline-color: {p.line};
}}
QTableView, QTableWidget {{ font-family: "{MONO}"; font-size: 12px; }}
QHeaderView::section {{
    background: {p.surface};
    color: {p.muted};
    border: none;
    border-bottom: 1px solid {p.line};
    border-right: 1px solid {p.line};
    padding: 4px 8px;
    font-family: "{SANS}";
    font-size: 11px;
}}
QTableView::item:selected, QListWidget::item:selected {{ background: {p.line}; color: {p.text}; }}

/* Text areas */
QPlainTextEdit, QTextEdit {{
    background: {p.bg};
    border: 1px solid {p.line};
    font-family: "{MONO}";
    font-size: 12px;
}}

/* Scrollbars */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {p.line_strong}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:vertical {{ margin: 2px 3px; }}
QScrollBar::handle:horizontal {{ margin: 3px 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* Menus */
QMenuBar {{ background: {p.bg}; border-bottom: 1px solid {p.line}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {p.raised}; }}
QMenu {{ background: {p.raised}; border: 1px solid {p.line_strong}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 12px; }}
QMenu::item:selected {{ background: {p.line}; }}
QMenu::separator {{ height: 1px; background: {p.line}; margin: 4px 6px; }}

QSplitter::handle {{ background: {p.line}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QProgressBar {{ background: {p.line}; border: none; max-height: 2px; }}
QProgressBar::chunk {{ background: {p.signal}; }}
QDialog {{ background: {p.bg}; }}
"""


def _icon_url(name: str, color: str) -> str:
    from .icons import cached_svg_path

    return cached_svg_path(name, color).as_posix()


def apply(app, palette_name: str = "console") -> Palette:
    """Apply a palette to the application and to matplotlib."""
    global _current
    _current = PALETTES.get(palette_name, CONSOLE)
    from PyQt6.QtGui import QFont

    font = QFont(SANS)
    font.setPixelSize(13)
    font.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    app.setFont(font)
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet(_current))
    return _current


def matplotlib_rc(p: Palette | None = None) -> dict:
    """rcParams that make matplotlib figures match the interface."""
    p = p or _current
    return {
        "figure.facecolor": p.bg,
        "axes.facecolor": p.bg,
        "savefig.facecolor": p.bg,
        "axes.edgecolor": p.line_strong,
        "axes.labelcolor": p.muted,
        "axes.titlecolor": p.text,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.prop_cycle": __import__("cycler").cycler(color=list(p.series)),
        "grid.color": p.line,
        "grid.linewidth": 0.6,
        "xtick.color": p.muted,
        "ytick.color": p.muted,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": p.muted,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "font.family": [SANS, "DejaVu Sans"],
        "font.size": 10,
        "text.color": p.text,
        "mathtext.default": "regular",
    }
