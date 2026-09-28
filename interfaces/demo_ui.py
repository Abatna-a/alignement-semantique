"""Interface Streamlit commune : thème clair, tableaux et clic pour localiser."""

from html import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

LIGHT_CSS = """
<style>
  html, body, [data-testid="stAppViewContainer"], .stApp,
  [data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
  }
  [data-testid="stSidebar"] {
    background-color: #f7f6f4 !important;
    color: #1a1a1a !important;
  }
  [data-testid="stSidebar"] * {
    color: #1a1a1a !important;
  }
  [data-testid="stMetric"] {
    background-color: #f7f6f4;
    border: 1px solid #e8e4df;
    border-radius: 8px;
    padding: 8px 12px;
  }
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #1a1a1a !important;
  }
  h1, h2, h3, h4, h5, p, label {
    color: #1a1a1a;
  }
  .stTabs [data-baseweb="tab-list"] {
    background-color: #ffffff;
  }
  [data-testid="stSelectbox"] label,
  [data-testid="stSelectbox"] p {
    color: #1a1a1a !important;
  }
  [data-testid="stSelectbox"] [data-baseweb="select"] > div,
  [data-testid="stSelectbox"] input {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border-color: #d0ccc6 !important;
  }
  [data-baseweb="popover"],
  [data-baseweb="menu"],
  [data-baseweb="popover"] ul,
  [data-baseweb="menu"] ul,
  ul[role="listbox"],
  [data-testid="stSelectboxVirtualDropdown"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
  }
  li[role="option"],
  [data-baseweb="menu"] li,
  [data-baseweb="popover"] li {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
  }
  li[role="option"]:hover,
  li[role="option"][aria-selected="true"],
  [data-baseweb="menu"] li:hover {
    background-color: #f3eee8 !important;
    color: #1a1a1a !important;
  }
  .loc-wrap {
    max-height: 650px;
    overflow: auto;
    border: 1px solid #d0ccc6;
    border-radius: 8px;
    background: #ffffff;
  }
  .loc-table {
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 13px;
    color: #1a1a1a;
    background: #ffffff;
  }
  .loc-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: #f3f0eb;
    color: #1a1a1a;
    font-weight: 700;
    text-align: left;
    padding: 9px 10px;
    border-bottom: 1px solid #c8c4be;
    white-space: nowrap;
  }
  .loc-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #eceae6;
    background: #ffffff;
    color: #1a1a1a;
    vertical-align: top;
  }
  .loc-table tbody tr:hover td {
    background: #f7f6f4;
  }
  .loc-mention {
    cursor: pointer;
    font-weight: 700;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .loc-mention.yellow { color: #b45309; }
  .loc-mention.red { color: #b91c1c; }
</style>
"""

TABLE_CLICK_CAPTION = (
    "Cliquez la mention : le span s’encadre tout de suite dans la note. "
    "Orange = ajout, rouge = manque. Aucun rechargement."
)

COLOR_STYLES = {
    "green": {"bg": "#28a745", "fg": "white", "label": "Match"},
    "yellow": {"bg": "#ffc107", "fg": "#1e293b", "label": "Ajout"},
    "red": {"bg": "#dc3545", "fg": "white", "label": "Manque"},
}

LOCATE_JS = """
<script>
(function () {
  function docs() {
    const out = [];
    let win = window;
    for (let i = 0; i < 8; i += 1) {
      try { out.push(win.document); } catch (err) {}
      if (!win.parent || win.parent === win) break;
      win = win.parent;
    }
    return out;
  }

  function findById(id) {
    function search(doc) {
      if (!doc) return null;
      const direct = doc.getElementById(id);
      if (direct) return direct;
      const frames = doc.querySelectorAll("iframe");
      for (let i = 0; i < frames.length; i += 1) {
        try {
          const hit = search(frames[i].contentDocument);
          if (hit) return hit;
        } catch (err) {}
      }
      return null;
    }
    for (const doc of docs()) {
      const hit = search(doc);
      if (hit) return hit;
    }
    return null;
  }

  function clearActive() {
    for (const doc of docs()) {
      doc.querySelectorAll(".loc-active").forEach(function (el) {
        el.classList.remove("loc-active");
        el.style.outline = "";
        el.style.outlineOffset = "";
      });
    }
  }

  function bind() {
    for (const doc of docs()) {
      doc.querySelectorAll(".loc-mention").forEach(function (el) {
        if (el.dataset.bound === "1") return;
        el.dataset.bound = "1";
        el.addEventListener("click", function (ev) {
          ev.preventDefault();
          ev.stopPropagation();
          const start = el.getAttribute("data-start");
          const end = el.getAttribute("data-end");
          const cat = el.getAttribute("data-cat");
          const color = cat === "yellow" ? "#e67e22" : "#dc3545";
          const target = findById("loc-" + start + "-" + end);
          if (!target) return;
          clearActive();
          target.classList.add("loc-active");
          target.style.outline = "3px solid " + color;
          target.style.outlineOffset = "2px";
          target.scrollIntoView({block: "center", inline: "nearest"});
        });
      });
    }
  }

  bind();
  setInterval(bind, 400);
})();
</script>
"""


def loc_id(start: int, end: int) -> str:
    return f"loc-{int(start)}-{int(end)}"


def color_span_html(text: str, category: str, start: int | None = None, end: int | None = None) -> str:
    style = COLOR_STYLES[category]
    loc = f' id="{loc_id(start, end)}"' if start is not None and end is not None else ""
    return (
        f'<span{loc} title="{style["label"]}" style="background-color: {style["bg"]}; '
        f'color: {style["fg"]}; padding: 2px 4px; border-radius: 4px; '
        f'cursor: help; font-weight: bold;">{text}</span>'
    )


def longest_wins(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    ordered = sorted(spans, key=lambda x: (x[1] - x[0]), reverse=True)
    kept: list[tuple[int, int, str]] = []
    for start, end, cat in ordered:
        if any(max(start, fs) < min(end, fe) for fs, fe, _ in kept):
            continue
        kept.append((start, end, cat))
    return kept


def paint_located_note(
    text: str,
    color_spans: list[tuple[int, int, str]],
    locate_spans: list[tuple[int, int, str]],
) -> str:
    """Conserve les couleurs et enveloppe chaque mention du tableau d'un id loc-start-end stable."""
    colors = longest_wins(color_spans)
    locates = []
    seen: set[tuple[int, int]] = set()
    for start, end, cat in locate_spans:
        key = (int(start), int(end))
        if key in seen or key[1] <= key[0] or key[0] < 0 or key[1] > len(text):
            continue
        seen.add(key)
        locates.append((key[0], key[1], cat))

    events: list[tuple[int, int, str, str, int, int]] = []
    for start, end, cat in colors:
        events.append((end, 1, "close_color", cat, start, end))
        events.append((start, 2, "open_color", cat, start, end))
    for start, end, cat in locates:
        events.append((end, 0, "close_loc", cat, start, end))
        events.append((start, 3, "open_loc", cat, start, end))
    events.sort(key=lambda item: (item[0], item[1]))

    parts: list[str] = []
    cursor = 0
    for pos, _prio, kind, cat, start, end in events:
        if pos > cursor:
            parts.append(text[cursor:pos])
            cursor = pos
        if kind == "open_color":
            style = COLOR_STYLES[cat]
            parts.append(
                f'<span title="{style["label"]}" style="background-color: {style["bg"]}; '
                f'color: {style["fg"]}; padding: 2px 4px; border-radius: 4px; '
                f'cursor: help; font-weight: bold;">'
            )
        elif kind == "close_color":
            parts.append("</span>")
        elif kind == "open_loc":
            parts.append(f'<span id="{loc_id(start, end)}" class="loc-target" data-cat="{cat}">')
        elif kind == "close_loc":
            parts.append("</span>")
    parts.append(text[cursor:])
    return "".join(parts)


def paint_char_located_note(
    text: str,
    color_spans: list[tuple[int, int, str]],
    locate_spans: list[tuple[int, int, str]],
) -> str:
    """Couleurs au caractère (PARHAF) et enveloppe loc-start-end par mention du tableau."""
    n = len(text)
    priority = {"red": 3, "yellow": 2, "green": 1}
    color: list[str | None] = [None] * n
    prio = [0] * n
    for start, end, cat in color_spans:
        rank = priority[cat]
        for i in range(max(0, start), min(n, end)):
            if rank > prio[i]:
                color[i] = cat
                prio[i] = rank

    opens: dict[int, list[tuple[int, str]]] = {}
    closes: dict[int, int] = {}
    seen: set[tuple[int, int]] = set()
    for start, end, cat in locate_spans:
        key = (int(start), int(end))
        if key in seen or key[1] <= key[0] or key[0] < 0 or key[1] > n:
            continue
        seen.add(key)
        opens.setdefault(key[0], []).append((key[1], cat))
        closes[key[1]] = closes.get(key[1], 0) + 1

    bounds = set(opens) | set(closes)
    parts: list[str] = []
    i = 0
    while i <= n:
        if i in closes:
            parts.append("</span>" * closes[i])
        if i == n:
            break
        if i in opens:
            for end, cat in opens[i]:
                parts.append(f'<span id="{loc_id(i, end)}" class="loc-target" data-cat="{cat}">')
        j = i + 1
        while j < n and color[j] == color[i] and j not in bounds:
            j += 1
        chunk = escape(text[i:j])
        parts.append(chunk if color[i] is None else color_span_html(chunk, color[i]))
        i = j
    return "".join(parts)


def apply_light_theme() -> None:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def want_detail_tables() -> bool:
    return st.checkbox(
        "Afficher le détail des prédictions (ajouts / manques)",
        value=False,
        help="Décoché : la note prend toute la largeur. Coché : tableaux à droite.",
    )


def render_note_panel(html_text: str, *, wide: bool = True, scroll_to_focus: bool = False) -> None:
    st.subheader("Analyse visuelle du texte")
    height_px = 780 if wide else 720
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: #ffffff; color: #1e293b; }}
  #note-panel {{
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px; line-height: 1.35;
    background: #ffffff; padding: 12px 16px;
  }}
</style></head>
<body>
<div id="note-panel">{html_text}</div>
</body></html>"""
    components.html(page, height=height_px, scrolling=True)


def install_locate_script() -> None:
    components.html(LOCATE_JS, height=0)


def focused_span(tables, note_id: str):
    """La localisation côté client ne relance pas Streamlit. Conservé pour les appels existants."""
    return None


def plan_with_focus(spans: list[tuple[int, int, str]], focus) -> list[tuple[int, int, str, bool]]:
    plan = [(s, e, c, False) for s, e, c in spans]
    plan.sort(key=lambda x: x[0], reverse=True)
    return plan


def focus_anchor(inner_html: str, category: str = "red") -> str:
    return inner_html


def render_clickable_table(
    df: pd.DataFrame,
    *,
    mention_col: str,
    category: str,
    columns: list[str],
) -> None:
    """Tableau léger dans un iframe : un clic ne navigue pas et ne relance pas Streamlit."""
    if df is None or df.empty:
        return
    view = df.reset_index(drop=True)
    header = "".join(f"<th>{escape(col)}</th>" for col in columns)
    rows: list[str] = []
    for data in view.to_dict("records"):
        start = int(data["start"])
        end = int(data["end"])
        cells: list[str] = []
        for col in columns:
            raw = data.get(col, "")
            text = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)
            if col == mention_col:
                cells.append(
                    "<td>"
                    f'<span class="loc-mention {category}" data-start="{start}" '
                    f'data-end="{end}" data-cat="{category}">{escape(text)}</span>'
                    "</td>"
                )
            else:
                cells.append(f"<td>{escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: #ffffff; }}
  .loc-wrap {{ height: 630px; overflow: auto; border: 1px solid #d0ccc6; border-radius: 8px; background: #ffffff; }}
  table {{ width: 100%; border-collapse: collapse; font-family: Arial, Helvetica, sans-serif; font-size: 13px; color: #1a1a1a; background: #ffffff; }}
  th {{ position: sticky; top: 0; background: #f3f0eb; color: #1a1a1a; font-weight: 700; text-align: left; padding: 9px 10px; border-bottom: 1px solid #c8c4be; white-space: nowrap; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eceae6; background: #ffffff; color: #1a1a1a; vertical-align: top; }}
  tr:hover td {{ background: #f7f6f4; }}
  .loc-mention {{ cursor: pointer; font-weight: 700; text-decoration: underline; text-underline-offset: 2px; }}
  .loc-mention.yellow {{ color: #b45309; }}
  .loc-mention.red {{ color: #b91c1c; }}
</style></head>
<body>
<div class="loc-wrap"><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{LOCATE_JS}
</body></html>"""
    components.html(page, height=650, scrolling=False)


SELECT_KWARGS = {}


def remember_table(table_key: str, note_id: str, spans) -> None:
    return None


def reset_clickable(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)
