import streamlit as st
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


class Match:
    def __init__(self, start, end, dist):
        self.start = start
        self.end = end
        self.dist = dist


def find_near_matches(pattern: str, text: str, max_l_dist: int):
    if not pattern or not text:
        return []

    matches = []
    pattern_len = len(pattern)

    for window_size in [pattern_len, pattern_len - 1, pattern_len + 1]:
        if window_size <= 0 or window_size > len(text):
            continue

        chunks = [
            (text[i : i + window_size], i) for i in range(len(text) - window_size + 1)
        ]
        if not chunks:
            continue

        result = process.extractOne(
            pattern, [c[0] for c in chunks], scorer=Levenshtein.distance
        )

        if result:
            matched_text, score, idx = result
            dist = int(score)
            if dist <= max_l_dist:
                start_pos = chunks[idx][1]
                matches.append(Match(start_pos, start_pos + window_size, dist))

    return matches


@st.cache_data
def highlight_text(test_obj, res):
    texts_to_highlight = []
    if hasattr(test_obj, "text"):
        texts_to_highlight.append(("text", test_obj.text))
    if hasattr(test_obj, "before"):
        texts_to_highlight.append(("before", test_obj.before))
    if hasattr(test_obj, "after"):
        texts_to_highlight.append(("after", test_obj.after))
    if hasattr(test_obj, "cell") and test_obj.cell:
        texts_to_highlight.append(("cell", test_obj.cell))
    if hasattr(test_obj, "up") and test_obj.up:
        texts_to_highlight.append(("up", test_obj.up))
    if hasattr(test_obj, "down") and test_obj.down:
        texts_to_highlight.append(("down", test_obj.down))
    if hasattr(test_obj, "left") and test_obj.left:
        texts_to_highlight.append(("left", test_obj.left))
    if hasattr(test_obj, "right") and test_obj.right:
        texts_to_highlight.append(("right", test_obj.right))
    if hasattr(test_obj, "top_heading") and test_obj.top_heading:
        texts_to_highlight.append(("top_heading", test_obj.top_heading))
    if hasattr(test_obj, "left_heading") and test_obj.left_heading:
        texts_to_highlight.append(("left_heading", test_obj.left_heading))

    matches_with_pos = []
    for label, txt in texts_to_highlight:
        if txt and txt.strip():
            fuzzy_matches = find_near_matches(
                txt, res, max_l_dist=min(20, len(txt) // 2)
            )
            for match in fuzzy_matches:
                matches_with_pos.append((match.start, match.end, label, match.dist))

    def remove_overlaps(matches):
        matches = sorted(matches, key=lambda x: (x[3], x[0]))
        result = []
        for match in matches:
            s1, e1, _, _ = match
            overlapping = False
            for s2, e2, _, _ in result:
                if not (e1 <= s2 or e2 <= s1):
                    overlapping = True
                    break
            if not overlapping:
                result.append(match)
        return result

    matches_with_pos = remove_overlaps(matches_with_pos)
    matches_with_pos.sort(key=lambda x: x[0], reverse=True)

    colors = [
        "yellow",
        "lightgreen",
        "lightblue",
        "lightcoral",
        "lightyellow",
        "lightpink",
        "lightgray",
        "lavender",
        "peachpuff",
        "palegreen",
    ]
    label_to_color = {}

    for start, end, label, dist in matches_with_pos:
        if label not in label_to_color:
            label_to_color[label] = colors[len(label_to_color) % len(colors)]
        color = label_to_color[label]
        res = (
            res[:start]
            + f'<span style="background-color: {color}; font-weight: bold;" title="{label}: edit distance={dist}">{res[start:end]}</span>'
            + res[end:]
        )
    return res
