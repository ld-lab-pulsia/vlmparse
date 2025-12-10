# from docling-core
"""Models for the base data types."""

import copy
import random
from typing import Literal, Tuple

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

FONT_SIZE = 20
FONT_CLASS_SIZE = 20
WIDTH_BOXES = 5


class Size(BaseModel):
    """Size."""

    width: float = 0.0
    height: float = 0.0

    def as_tuple(self):
        return (self.width, self.height)


def clip(val, minimum, maximum):
    return max(min(val, maximum), minimum)


CoordOrigin = Literal["TOPLEFT", "BOTTOMLEFT"]

AngleDirection = Literal["COUNTERCLOCKWISE"]  ### "CLOCKWISE" is not supported !

AngleType = Literal["DEGREES"]  ### "RADIANS" is not supported !


class InvalidBoundingBoxCoordinates(PydanticCustomError):
    pass


class BoundingBox(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=True,
    )
    """BoundingBox."""

    l: float = Field(ge=0)  # left
    t: float = Field(ge=0)  # top
    r: float = Field(ge=0)  # right
    b: float = Field(ge=0)  # bottom

    coord_origin: CoordOrigin = "TOPLEFT"
    relative: bool = False
    reference: Size | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_bbox(cls, field_values):
        if "coord_origin" not in field_values:
            field_values["coord_origin"] = "TOPLEFT"
        cond = True
        if field_values["coord_origin"] == "TOPLEFT":
            cond &= field_values["l"] < field_values["r"]
            cond &= field_values["t"] < field_values["b"]
        elif field_values["coord_origin"] == "BOTTOMLEFT":
            cond &= field_values["l"] < field_values["r"]
            cond &= field_values["t"] > field_values["b"]
        if not cond:
            raise InvalidBoundingBoxCoordinates(
                "InvalidBoundingBoxCoordinates",
                f"Invalid BoundingBox coordinates: left={field_values['l']}, top={field_values['t']}, right={field_values['r']}, bottom={field_values['b']}",
            )
        return field_values

    @property
    def width(self):
        """width."""
        return self.r - self.l

    @property
    def height(self):
        """height."""
        return abs(self.t - self.b)

    @property
    def center(self):
        """center"""
        return (float(self.r - self.l) / 2.0, float(self.b - self.t) / 2.0)

    def scaled(self, scale: float) -> "BoundingBox":
        """scaled.

        :param scale: float:

        """
        out_bbox = self.model_dump()
        out_bbox["l"] *= scale  # noqa
        out_bbox["r"] *= scale
        out_bbox["t"] *= scale
        out_bbox["b"] *= scale

        return BoundingBox.model_validate(out_bbox)

    def normalized(self, page_size: Size) -> "BoundingBox":
        """normalized.

        :param page_size: Size:

        """
        out_bbox = self.model_dump()
        out_bbox["l"] /= page_size.width  # noqa
        out_bbox["r"] /= page_size.width
        out_bbox["t"] /= page_size.height
        out_bbox["b"] /= page_size.height
        out_bbox["relative"] = True
        out_bbox["reference"] = page_size
        return BoundingBox.model_validate(out_bbox)

    def unormalized(self, page_size: Size) -> "BoundingBox":
        """normalized.

        :param page_size: Size:

        """
        out_bbox = self.model_dump()
        out_bbox["l"] *= page_size.width  # noqa
        out_bbox["r"] *= page_size.width
        out_bbox["t"] *= page_size.height
        out_bbox["b"] *= page_size.height
        out_bbox["relative"] = False
        out_bbox["reference"] = None
        return BoundingBox.model_validate(out_bbox)

    def to_relative(self, page_size: Size, inplace: bool = False) -> "BoundingBox":
        """to_relative. INPLACE

        :param page_size: Size:

        """
        if inplace:
            if self.relative:
                return self
            assert self.reference is None
            self = self.normalized(page_size)
            return self
        else:
            out_bbox = copy.deepcopy(self)
            if out_bbox.relative:
                return out_bbox
            assert out_bbox.reference is None
            out_bbox = out_bbox.normalized(page_size)
            return out_bbox

    def to_absolute(self, page_size: Size, inplace: bool = False) -> "BoundingBox":
        """to_absolute. INPLACE

        :param page_size: Size:

        """
        if inplace:
            if not self.relative:
                return self
            assert self.reference is not None
            self = self.unormalized(page_size)
            return self
        else:
            if not self.relative:
                return BoundingBox.model_validate(self.model_dump())
            out_bbox = self.unormalized(page_size)
            return out_bbox

    def clip(self, page_size: Size) -> "BoundingBox":
        out_bbox = copy.deepcopy(self)
        out_bbox.l = clip(out_bbox.l, 0, page_size.width)
        out_bbox.r = clip(out_bbox.r, 0, page_size.width)
        out_bbox.t = clip(out_bbox.t, 0, page_size.height)
        out_bbox.b = clip(out_bbox.b, 0, page_size.height)
        return out_bbox

    def get_relative_box(self, reference_box: "BoundingBox") -> "BoundingBox":
        l = max(0, self.l - reference_box.l)
        r = max(0, self.r - reference_box.l)

        if self.coord_origin == "TOPLEFT":
            t = max(0, self.t - reference_box.t)
            b = max(0, self.b - reference_box.t)
        else:  # BOTTOMLEFT
            t = max(0, self.t - reference_box.b)
            b = max(0, self.b - reference_box.b)

        return type(self)(l=l, t=t, r=r, b=b, coord_origin=self.coord_origin)

    def as_tuple(self, output_format: Literal["TOPLEFT", "BOTTOMLEFT"] = "TOPLEFT"):
        """as_tuple."""
        if output_format == "TOPLEFT":
            return (self.l, self.t, self.r, self.b)
        elif output_format == "BOTTOMLEFT":
            return (self.l, self.b, self.r, self.t)

    def as_tuple_xywh(self):
        return (self.l, self.t, self.width, self.height)

    @classmethod
    def from_tuple(cls, coord: Tuple[float, ...], origin: CoordOrigin):
        """from_tuple.

        :param coord: Tuple[float:
        :param ...]:
        :param origin: CoordOrigin:

        """
        if origin == "TOPLEFT":
            l, t, r, b = coord[0], coord[1], coord[2], coord[3]
            if r < l:
                l, r = r, l
            if b < t:
                b, t = t, b

            return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=origin)
        elif origin == "BOTTOMLEFT":
            l, b, r, t = coord[0], coord[1], coord[2], coord[3]
            if r < l:
                l, r = r, l
            if b > t:
                b, t = t, b

            return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=origin)

    def area(self) -> float:
        """area."""
        area = (self.r - self.l) * (self.b - self.t)
        if self.coord_origin == "BOTTOMLEFT":
            area = -area
        return area

    def intersection_area_with(self, other: "BoundingBox") -> float:
        """intersection_area_with.

        :param other: "BoundingBox":

        """
        # Calculate intersection coordinates
        left = max(self.l, other.l)
        top = max(self.t, other.t)
        right = min(self.r, other.r)
        bottom = min(self.b, other.b)

        # Calculate intersection dimensions
        width = right - left
        height = bottom - top

        # If the bounding boxes do not overlap, width or height will be negative
        if width <= 0 or height <= 0:
            return 0.0

        return width * height

    def intersection_over_self_area(self, other: "BoundingBox") -> float:
        """intersection_over_self_area.

        :param other: "BoundingBox":

        """
        intersection_area = self.intersection_area_with(other)
        return intersection_area / self.area()

    def intersection_over_union(self, other: "BoundingBox") -> float:
        """intersection_over_union.

        :param other: "BoundingBox":

        """
        intersection_area = self.intersection_area_with(other)
        union_area = self.area() + other.area() - intersection_area
        return intersection_area / union_area

    def intersection_over_minimum(self, other: "BoundingBox") -> float:
        """intersection_over_minimum.

        :param other: "BoundingBox":

        """
        intersection_area = self.intersection_area_with(other)
        return intersection_area / min(self.area(), other.area())

    def vertical_distance(self, box: "BoundingBox") -> float:
        return min(abs(box.t - self.b), abs(self.t - box.b))

    def horizontal_distance(self, box: "BoundingBox") -> float:
        return min(abs(box.t - self.b), abs(self.t - box.b))

    def is_not_too_low(self, box: "BoundingBox", threshold: float = 0.01) -> bool:
        return max(self.b - box.b, 0) <= threshold * self.height

    def is_not_too_high(self, box: "BoundingBox", threshold: float = 0.01) -> bool:
        return max(box.t - self.t, 0) <= threshold * self.height

    def is_inside(self, box: "BoundingBox", threshold: float = 0.2) -> bool:
        cond_left = False
        if self.l >= box.l:
            cond_left = abs(self.t - box.t) <= threshold * self.height
        else:
            cond_left = True
        cond_right = False
        if self.r <= box.r:
            cond_right = abs(self.r - box.r) <= threshold * self.height
        else:
            cond_right = True
        return cond_left and cond_right

    def to_bottom_left_origin(self, page_height) -> "BoundingBox":
        """to_bottom_left_origin.

        :param page_height:

        """
        if self.coord_origin == "BOTTOMLEFT":
            return self
        elif self.coord_origin == "TOPLEFT":
            return BoundingBox(
                l=self.l,
                r=self.r,
                t=page_height - self.t,
                b=page_height - self.b,
                coord_origin="BOTTOMLEFT",
            )

    def to_top_left_origin(self, page_height):
        """to_top_left_origin.

        :param page_height:

        """
        if self.coord_origin == "TOPLEFT":
            return self
        elif self.coord_origin == "BOTTOMLEFT":
            return BoundingBox(
                l=self.l,
                r=self.r,
                t=page_height - self.t,  # self.b
                b=page_height - self.b,  # self.t
                coord_origin="TOPLEFT",
            )

    def plot(
        self,
        image: Image.Image,
        color: tuple | None = (255, 0, 0),
        width: int = WIDTH_BOXES,
    ):
        box = self.to_top_left_origin(image.size[1])
        if self.relative:
            box_absolute = box.to_absolute(page_size=self.reference, inplace=False)
        else:
            box_absolute = box
        if color is None:
            color = tuple(random.sample(range(0, 255, 1), 3))
        draw = ImageDraw.Draw(image)

        draw.rectangle(
            box_absolute.as_tuple(),
            outline=color,
            width=width,
        )
        return image

    @staticmethod
    def merge_boxes(boxes: list["BoundingBox"]) -> "BoundingBox":
        list_x = [box.l for box in boxes] + [box.r for box in boxes]
        list_y = [box.t for box in boxes] + [box.b for box in boxes]
        for b in boxes:
            assert b.coord_origin == boxes[0].coord_origin
            assert b.relative == boxes[0].relative
            assert b.reference == boxes[0].reference
        return BoundingBox(
            coord_origin=boxes[0].coord_origin,
            relative=boxes[0].relative,
            reference=boxes[0].reference,
            l=min(list_x),
            r=max(list_x),
            t=min(list_y),
            b=max(list_y),
        )

    def rotate(self, angle: Literal[90, 180, 270], size: Size) -> "BoundingBox":
        original_relative = self.relative
        if self.coord_origin != "TOPLEFT":
            raise ValueError("rotate assumes TOPLEFT coordinate origin")

        # convert to absolute coordinates if needed
        if original_relative:
            l = self.l * size.width
            t = self.t * size.height
            r = self.r * size.width
            b = self.b * size.height
        else:
            l, t, r, b = self.l, self.t, self.r, self.b

        width, height = size.width, size.height

        if angle == 270:
            dx, dy = height, 0
        elif angle == 180:
            dx, dy = width, height
        else:  # 90
            dx, dy = 0, width

        corners = [
            (l, t),
            (r, t),
            (r, b),
            (l, b),
        ]

        def _rotate_point(x: float, y: float):
            if angle == 270:
                return -y, x
            elif angle == 180:
                return -x, -y
            else:  # 90
                return y, -x

        rotated = [_rotate_point(x, y) for x, y in corners]
        xs = [dx + p[0] for p in rotated]
        ys = [dy + p[1] for p in rotated]

        l_new, r_new = min(xs), max(xs)
        t_new, b_new = min(ys), max(ys)

        if angle in (90, 270):
            new_width, new_height = height, width
        else:
            new_width, new_height = width, height

        if original_relative:
            return BoundingBox(
                l=l_new / new_width,
                t=t_new / new_height,
                r=r_new / new_width,
                b=b_new / new_height,
                coord_origin="TOPLEFT",
                relative=True,
                reference=Size(width=new_width, height=new_height),
            )
        else:
            return BoundingBox(
                l=l_new,
                t=t_new,
                r=r_new,
                b=b_new,
                coord_origin="TOPLEFT",
                relative=False,
                reference=None,
            )


def get_width_height_text(text, font, font_size):
    size_text = font.getmask(text).getbbox()
    if size_text is None:
        (text_width, text_height) = (0, font_size)
    else:
        (text_width, text_height) = size_text[2::]
    return text_width, text_height + 5


def draw_highlighted_text(
    image,
    rectangle_coords,
    text,
    text_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    opacity=0.8,
    font_size: int = 20,
    font_name: str = "DejaVuSans.ttf",
):
    """
    Dessine un texte avec un rectangle semi-transparent en surbrillance.

    Args:
        image (PIL.Image.Image): L'image de base.
        rectangle_coords (tuple): Les coordonnées du rectangle (x1, y1, x2, y2).
        text (str): Le texte à écrire.
        text_color (tuple): Couleur du texte (R, G, B).
        bg_color (tuple): Couleur de fond du rectangle (R, G, B).
        opacity (float): Opacité du rectangle (entre 0 et 1).

    Returns:
        PIL.Image.Image: L'image modifiée.
    """
    font = ImageFont.truetype(font_name, font_size)
    # Créer une image semi-transparente pour le rectangle
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Extraire les coordonnées
    x1, y1, x2, y2 = rectangle_coords

    # Dessiner le rectangle semi-transparent
    bg_color_with_opacity = (
        *bg_color,
        int(255 * opacity),
    )  # Ajouter l'opacité à la couleur de fond
    draw_overlay.rectangle(
        rectangle_coords, fill=bg_color_with_opacity, width=WIDTH_BOXES
    )

    # Superposer l'overlay sur l'image de base
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)

    # Créer un objet ImageDraw sur l'image combinée
    draw_combined = ImageDraw.Draw(combined)

    # Calculer la position centrée du texte

    (text_width, text_height) = get_width_height_text(text, font, font_size)
    text_x = x1 + (x2 - x1 - text_width) // 2
    text_y = y1 + (y2 - y1 - text_height) // 2

    # Dessiner le texte par-dessus le rectangle
    draw_combined.text((text_x, text_y), text, fill=text_color, font=font)

    return combined.convert("RGB")  # Convertir en mode RGB si besoin


def draw_text_of_box(
    image,
    left,
    top,
    text,
    font_size: int = 20,
    font_name: str = "DejaVuSans.ttf",
    text_inside: bool = True,
    **kwargs,
):
    font = ImageFont.truetype(font_name, font_size)
    (text_width, text_height) = get_width_height_text(text, font, font_size)
    if text_inside:
        rectangle_coords = (left, top, left + text_width, top + text_height)
    else:
        rectangle_coords = (left, top - text_height, left + text_width, top)
    return draw_highlighted_text(
        image=image,
        rectangle_coords=rectangle_coords,
        text=text,
        font_name=font_name,
        font_size=font_size,
        **kwargs,
    )
