from pathlib import Path

from PIL import Image, ImageDraw


def _draw_thirds(draw: ImageDraw.ImageDraw, w: int, h: int):
    color = (180, 180, 180)
    draw.line([(w // 3, 0), (w // 3, h)], fill=color, width=1)
    draw.line([(2 * w // 3, 0), (2 * w // 3, h)], fill=color, width=1)
    draw.line([(0, h // 3), (w, h // 3)], fill=color, width=1)
    draw.line([(0, 2 * h // 3), (w, 2 * h // 3)], fill=color, width=1)


def draw_scoring_overlay(image_rgb: Image.Image, predicted_score: float):
    output = image_rgb.copy()
    draw = ImageDraw.Draw(output)

    w, h = output.size
    _draw_thirds(draw, w, h)

    text_lines = [
        "Predicted Score: {:.2f}/10".format(predicted_score),
    ]
    y = 12
    for line in text_lines:
        draw.text((12, y), line, fill=(255, 255, 255))
        y += 18

    return output


def save_visualization(image_rgb: Image.Image, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb.save(str(out_path))
