import base64
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image


def vectorized_bootstrap_grouped_std(df, group_col, value_col, n_bootstrap=1000):
    group_col = [group_col] if isinstance(group_col, str) else group_col
    grouped = df.groupby(group_col)[value_col]

    def bootstrap_group(group):
        values = group.values
        n = len(values)
        bootstrap_samples = np.random.choice(
            values, size=(n_bootstrap, n), replace=True
        )
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        return pd.Series(
            {"mean": np.mean(values), "bootstrap_std": np.std(bootstrap_means)}
        )

    result = grouped.apply(bootstrap_group)
    return result.unstack(-1)


def format_results_vectorized(result_df, precision=2):
    means = result_df["mean"].values
    margins = 2 * result_df["bootstrap_std"].values

    formatted = np.char.add(
        np.char.add(np.round(means, precision).astype(str), " ± "),
        np.round(margins, precision).astype(str),
    )

    return pd.DataFrame({"formatted_result": formatted}, index=result_df.index)


def bootstrap_and_format_results(
    df, group_col, value_col, n_bootstrap=1000, precision=2
):
    result_df = vectorized_bootstrap_grouped_std(df, group_col, value_col, n_bootstrap)
    return format_results_vectorized(result_df, precision)


def to_base64(image: Image, extension="PNG"):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=extension)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")


def from_base64(base64_str: str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))
