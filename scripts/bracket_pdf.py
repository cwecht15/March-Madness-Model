from __future__ import annotations

import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)
MARGIN = 24
HEADER_Y = PAGE_HEIGHT - 28
SUBHEADER_Y = PAGE_HEIGHT - 42
BOX_HEIGHT = 24
BOX_RADIUS = 6
BOX_PADDING = 5
GAME_GAP = 6

REGION_COLUMNS = [26, 212, 390, 548]
REGION_WIDTHS = [158, 140, 124, 108]
FINAL_FOUR_COLUMNS = [82, 310, 520]
FINAL_FOUR_WIDTHS = [150, 140, 150]

ABBREVIATIONS = {
    "State": "St.",
    "Saint": "St.",
    "Mount": "Mt.",
    "Northern": "N.",
    "Southern": "S.",
    "Eastern": "E.",
    "Western": "W.",
    "Central": "C.",
    "Southeastern": "SE",
    "Southwestern": "SW",
    "Northeastern": "NE",
    "Northwestern": "NW",
    "University": "Univ.",
    "College": "Col.",
}


def _fit_text(canv: canvas.Canvas, text: str, font_name: str, max_size: float, max_width: float) -> float:
    size = max_size
    while size > 5.25 and stringWidth(text, font_name, size) > max_width:
        size -= 0.25
    return size


def _compact_team_label(team_label: str) -> str:
    parts = team_label.split()
    compacted = [ABBREVIATIONS.get(part, part) for part in parts]
    return " ".join(compacted)


def _ellipsize_text(
    canv: canvas.Canvas,
    text: str,
    font_name: str,
    font_size: float,
    max_width: float,
) -> str:
    if stringWidth(text, font_name, font_size) <= max_width:
        return text

    base = text.rstrip(". ")
    suffix = "..."
    while base and stringWidth(base + suffix, font_name, font_size) > max_width:
        base = base[:-1].rstrip()
    return (base + suffix) if base else suffix


def _format_label(team_label: str) -> str:
    return _compact_team_label(team_label or "TBD")


def _draw_page_header(canv: canvas.Canvas, title: str, subtitle: str | None = None) -> None:
    canv.setTitle("March Madness Bracket Picks")
    canv.setFillColor(colors.HexColor("#111827"))
    canv.setFont("Helvetica-Bold", 20)
    canv.drawString(MARGIN, HEADER_Y, title)
    if subtitle:
        canv.setFont("Helvetica", 9)
        canv.setFillColor(colors.HexColor("#4b5563"))
        canv.drawString(MARGIN, SUBHEADER_Y, subtitle)


def _draw_round_headers(canv: canvas.Canvas, headers: list[str], columns: list[float], widths: list[float]) -> None:
    canv.setFillColor(colors.HexColor("#111827"))
    canv.setFont("Helvetica-Bold", 10)
    for title, x, width in zip(headers, columns, widths):
        canv.drawCentredString(x + (width / 2.0), PAGE_HEIGHT - 64, title)


def _draw_team_box(
    canv: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    team_label: str,
    picked: bool,
) -> tuple[float, float]:
    fill = colors.white if picked else colors.HexColor("#f8fafc")
    stroke = colors.HexColor("#111827") if picked else colors.HexColor("#cbd5e1")
    font_name = "Helvetica-Bold" if picked else "Helvetica"
    max_text_width = width - (2 * BOX_PADDING)
    label = _format_label(team_label)
    font_size = _fit_text(canv, label, font_name, 8.8, max_text_width)
    fitted_label = _ellipsize_text(canv, label, font_name, font_size, max_text_width)

    canv.setFillColor(fill)
    canv.setStrokeColor(stroke)
    canv.setLineWidth(1.2 if picked else 0.7)
    canv.roundRect(x, y - BOX_HEIGHT + 2, width, BOX_HEIGHT, BOX_RADIUS, fill=1, stroke=1)

    canv.setFont(font_name, font_size)
    canv.setFillColor(colors.HexColor("#111827"))
    canv.drawString(x + BOX_PADDING, y - 14, fitted_label)
    return (x + width, y - (BOX_HEIGHT / 2.0))


def _draw_matchup_boxes(
    canv: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    top_label: str,
    bottom_label: str,
    top_picked: bool,
    bottom_picked: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    top_center = _draw_team_box(canv, x, y, width, top_label, top_picked)
    bottom_center = _draw_team_box(canv, x, y - BOX_HEIGHT, width, bottom_label, bottom_picked)
    return top_center, bottom_center


def _draw_connector(canv: canvas.Canvas, start: tuple[float, float], end: tuple[float, float]) -> None:
    canv.setStrokeColor(colors.HexColor("#94a3b8"))
    canv.setLineWidth(0.8)
    elbow_x = start[0] + ((end[0] - start[0]) * 0.45)
    canv.line(start[0], start[1], elbow_x, start[1])
    canv.line(elbow_x, start[1], elbow_x, end[1])
    canv.line(elbow_x, end[1], end[0], end[1])


def _round_one_positions(start_y: float) -> list[float]:
    positions: list[float] = []
    cursor = start_y
    for _ in range(8):
        positions.append(cursor)
        cursor -= (BOX_HEIGHT * 2) + GAME_GAP
    return positions


def _next_round_positions(previous: list[float]) -> list[float]:
    return [((previous[index] + previous[index + 1]) / 2.0) - 8 for index in range(0, len(previous), 2)]


def _build_region_rounds(
    games: dict[str, dict[str, Any]],
    order: list[str],
    picks: dict[str, str],
    seed_lookup: dict[str, int],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any] | None]:
    winners: dict[str, str] = {}
    regions: dict[str, dict[str, list[dict[str, Any]]]] = {
        region: {"round1": [], "round2": [], "round3": [], "round4": []}
        for region in ["East", "West", "South", "Midwest"]
    }
    semifinals: list[dict[str, Any]] = []
    championship: dict[str, Any] | None = None

    def label(team: str | None) -> str:
        if not team:
            return "TBD"
        seed = seed_lookup.get(team)
        return f"{seed} {team}" if seed is not None else team

    def resolve(source: tuple[str, str]) -> str | None:
        source_type, source_value = source
        if source_type == "team":
            return source_value
        return winners.get(source_value)

    for game_id in order:
        game = games[game_id]
        left_team = resolve(game["left_source"])
        right_team = resolve(game["right_source"])
        picked_winner = picks.get(game_id)
        if picked_winner:
            winners[game_id] = picked_winner

        round_index = int(game["round_index"])
        if round_index == 0:
            continue
        if round_index == 1:
            regions[str(game["region"])]["round1"].append(
                {
                    "left_label": label(left_team),
                    "right_label": label(right_team),
                    "picked": bool(picked_winner),
                    "winner_label": label(picked_winner) if picked_winner else None,
                }
            )
        elif round_index in {2, 3, 4}:
            key = {2: "round2", 3: "round3", 4: "round4"}[round_index]
            regions[str(game["region"])][key].append(
                {
                    "left_label": label(left_team),
                    "right_label": label(right_team),
                    "winner_label": label(picked_winner or left_team or right_team),
                    "picked": bool(picked_winner),
                }
            )
        elif round_index == 5:
            semifinals.append(
                {
                    "left_label": label(left_team),
                    "right_label": label(right_team),
                    "winner_label": label(picked_winner or left_team or right_team),
                    "picked": bool(picked_winner),
                    "label": str(game["label"]),
                }
            )
        elif round_index == 6:
            championship = {
                "left_label": label(left_team),
                "right_label": label(right_team),
                "winner_label": label(picked_winner or left_team or right_team),
                "picked": bool(picked_winner),
                "label": str(game["label"]),
            }

    return regions, semifinals, championship


def _draw_region_page(canv: canvas.Canvas, region_name: str, region_data: dict[str, list[dict[str, Any]]]) -> None:
    _draw_page_header(
        canv,
        f"{region_name} Region",
        "Bracket picks export from the interactive bracket builder",
    )
    _draw_round_headers(
        canv,
        ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"],
        REGION_COLUMNS,
        REGION_WIDTHS,
    )

    round1 = _round_one_positions(PAGE_HEIGHT - 92)
    round2 = _next_round_positions(round1)
    round3 = _next_round_positions(round2)
    round4 = _next_round_positions(round3)

    previous_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index, game in enumerate(region_data["round1"]):
        y = round1[index]
        picked_winner = game.get("winner_label")
        top_center, bottom_center = _draw_matchup_boxes(
            canv,
            REGION_COLUMNS[0],
            y,
            REGION_WIDTHS[0],
            game["left_label"],
            game["right_label"],
            bool(game["picked"] and picked_winner == game["left_label"]),
            bool(game["picked"] and picked_winner == game["right_label"]),
        )
        previous_pairs.append((top_center, bottom_center))

    for round_name, positions, x, width in [
        ("round2", round2, REGION_COLUMNS[1], REGION_WIDTHS[1]),
        ("round3", round3, REGION_COLUMNS[2], REGION_WIDTHS[2]),
        ("round4", round4, REGION_COLUMNS[3], REGION_WIDTHS[3]),
    ]:
        next_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for index, game in enumerate(region_data[round_name]):
            top_center, bottom_center = _draw_matchup_boxes(
                canv,
                x,
                positions[index],
                width,
                game["left_label"],
                game["right_label"],
                bool(game["picked"] and game["winner_label"] == game["left_label"]),
                bool(game["picked"] and game["winner_label"] == game["right_label"]),
            )
            _draw_connector(canv, previous_pairs[index][0], top_center)
            _draw_connector(canv, previous_pairs[index][1], bottom_center)
            next_pairs.append((top_center, bottom_center))
        previous_pairs = next_pairs

    canv.showPage()


def _draw_first_four_page(
    canv: canvas.Canvas,
    games: dict[str, dict[str, Any]],
    order: list[str],
    picks: dict[str, str],
    seed_lookup: dict[str, int],
) -> None:
    first_four_games = [game_id for game_id in order if int(games[game_id]["round_index"]) == 0]
    if not first_four_games:
        return

    def label(team: str) -> str:
        seed = seed_lookup.get(team)
        return f"{seed} {team}" if seed is not None else team

    _draw_page_header(canv, "First Four", "Play-in games listed separately for readability")
    canv.setFont("Helvetica-Bold", 11)
    canv.setFillColor(colors.HexColor("#111827"))

    y = PAGE_HEIGHT - 94
    for game_id in first_four_games:
        game = games[game_id]
        picked = picks.get(game_id)
        box_fill = colors.white if picked else colors.HexColor("#f8fafc")
        box_stroke = colors.HexColor("#111827") if picked else colors.HexColor("#cbd5e1")
        canv.setFillColor(box_fill)
        canv.setStrokeColor(box_stroke)
        canv.setLineWidth(1.0)
        canv.roundRect(MARGIN, y - 42, PAGE_WIDTH - (2 * MARGIN), 52, 8, fill=1, stroke=1)

        left_team = game["left_source"][1]
        right_team = game["right_source"][1]
        canv.setFillColor(colors.HexColor("#111827"))
        canv.setFont("Helvetica-Bold", 11)
        canv.drawString(MARGIN + 12, y - 10, str(game["label"]))
        canv.setFont("Helvetica", 10)
        canv.drawString(MARGIN + 12, y - 26, f"{_format_label(label(left_team))} vs {_format_label(label(right_team))}")
        if picked:
            canv.setFont("Helvetica-Bold", 10)
            canv.drawString(MARGIN + 12, y - 40, f"Picked winner: {_format_label(label(picked))}")
        y -= 76

    canv.showPage()


def _draw_final_four_page(
    canv: canvas.Canvas,
    semifinals: list[dict[str, Any]],
    championship: dict[str, Any] | None,
) -> None:
    _draw_page_header(canv, "Final Four", "National semifinal and championship picks")
    _draw_round_headers(
        canv,
        ["Semifinal 1", "Championship", "Semifinal 2"],
        FINAL_FOUR_COLUMNS,
        FINAL_FOUR_WIDTHS,
    )

    semi_top_y = PAGE_HEIGHT - 170
    semi_bottom_y = PAGE_HEIGHT - 270
    title_y = PAGE_HEIGHT - 220

    left_winner = "TBD"
    right_winner = "TBD"

    if len(semifinals) >= 1:
        semifinal = semifinals[0]
        top, bottom = _draw_matchup_boxes(
            canv,
            FINAL_FOUR_COLUMNS[0],
            semi_top_y,
            FINAL_FOUR_WIDTHS[0],
            semifinal["left_label"],
            semifinal["right_label"],
            bool(semifinal["picked"] and semifinal["winner_label"] == semifinal["left_label"]),
            bool(semifinal["picked"] and semifinal["winner_label"] == semifinal["right_label"]),
        )
        left_center = _draw_team_box(
            canv,
            FINAL_FOUR_COLUMNS[1],
            title_y,
            FINAL_FOUR_WIDTHS[1],
            semifinal["winner_label"],
            semifinal["picked"],
        )
        _draw_connector(canv, top, left_center)
        _draw_connector(canv, bottom, left_center)
        left_winner = semifinal["winner_label"]

    if len(semifinals) >= 2:
        semifinal = semifinals[1]
        top, bottom = _draw_matchup_boxes(
            canv,
            FINAL_FOUR_COLUMNS[2],
            semi_bottom_y,
            FINAL_FOUR_WIDTHS[2],
            semifinal["left_label"],
            semifinal["right_label"],
            bool(semifinal["picked"] and semifinal["winner_label"] == semifinal["left_label"]),
            bool(semifinal["picked"] and semifinal["winner_label"] == semifinal["right_label"]),
        )
        right_center = _draw_team_box(
            canv,
            FINAL_FOUR_COLUMNS[1],
            title_y - 74,
            FINAL_FOUR_WIDTHS[1],
            semifinal["winner_label"],
            semifinal["picked"],
        )
        _draw_connector(canv, top, right_center)
        _draw_connector(canv, bottom, right_center)
        right_winner = semifinal["winner_label"]

    title_top, title_bottom = _draw_matchup_boxes(
        canv,
        FINAL_FOUR_COLUMNS[1] - 120,
        PAGE_HEIGHT - 110,
        130,
        left_winner,
        right_winner,
        bool(championship and championship["picked"] and championship["winner_label"] == left_winner),
        bool(championship and championship["picked"] and championship["winner_label"] == right_winner),
    )
    champion_center = _draw_team_box(
        canv,
        FINAL_FOUR_COLUMNS[1],
        PAGE_HEIGHT - 110,
        FINAL_FOUR_WIDTHS[1],
        championship["winner_label"] if championship else "TBD",
        bool(championship and championship["picked"]),
    )
    _draw_connector(canv, title_top, champion_center)
    _draw_connector(canv, title_bottom, champion_center)

    canv.showPage()


def generate_bracket_pdf(
    games: dict[str, dict[str, Any]],
    order: list[str],
    picks: dict[str, str],
    seed_lookup: dict[str, int],
) -> bytes:
    buffer = io.BytesIO()
    canv = canvas.Canvas(buffer, pagesize=landscape(letter))

    regions, semifinals, championship = _build_region_rounds(games, order, picks, seed_lookup)

    _draw_first_four_page(canv, games, order, picks, seed_lookup)
    for region_name in ["East", "South", "West", "Midwest"]:
        _draw_region_page(canv, region_name, regions[region_name])
    _draw_final_four_page(canv, semifinals, championship)

    canv.save()
    buffer.seek(0)
    return buffer.getvalue()
