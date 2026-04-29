# svg_utils.py
import re


def clean_svg(svg_text):
    # 1. Remove XML comments
    # svg_text = re.sub(r"", "", svg_text, flags=re.DOTALL)

    def clean_path_data(match):
        path_string = match.group(1)
        path_string = re.sub(r"([a-zA-Z])", r" \1 ", path_string)
        path_string = path_string.replace(",", " ")

        def round_match(m):
            try:
                return f"{float(m.group(0)):.1f}"
            except ValueError:
                return m.group(0)

        path_string = re.sub(r"-?\d*\.?\d+", round_match, path_string)
        path_string = re.sub(r"\s+", " ", path_string).strip()
        return f'd="{path_string}"'

    svg_text = re.sub(r'd="([^"]+)"', clean_path_data, svg_text)
    svg_text = re.sub(r"\s+", " ", svg_text).strip()
    return svg_text


def process_row(example):
    example["Svg"] = clean_svg(example["Svg"])
    return example
