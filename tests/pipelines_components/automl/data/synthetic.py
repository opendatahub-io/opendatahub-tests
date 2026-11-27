"""Generate small, reproducible CSV datasets for AutoML smoke runs."""

import csv
from io import StringIO


def synthetic_automl_csv(task_type: str) -> str:
    """Generate 150 unique rows with the columns expected by the chosen pipeline.

    Args:
        task_type: Regression, classification, multiclass, or timeseries.

    Returns:
        CSV training data for the selected task type.

    Raises:
        ValueError: If the task type is unknown.
    """
    output = StringIO(newline="")
    writer = csv.writer(output)  # noqa: FCN001 - csvfile is positional-only

    if task_type == "timeseries":
        writer.writerow(["series_id", "Month", "Sunspots"])  # noqa: FCN001 - row is positional-only
        for row_index in range(150):
            month = f"{2010 + row_index // 12}-{row_index % 12 + 1:02d}-01"
            sunspots = round(35 + 0.4 * row_index + 1.7 * (row_index % 12), 2)
            writer.writerow(["sunspots", month, sunspots])  # noqa: FCN001 - row is positional-only
    elif task_type in ("regression", "classification", "multiclass"):
        label_column = "price" if task_type == "regression" else "target"
        writer.writerow(["row_id", "feature_a", "feature_b", label_column])  # noqa: FCN001 - row is positional-only
        for row_index in range(150):
            if task_type == "regression":
                label = round(100 + 2.75 * row_index + 3.2 * (row_index % 11), 2)
            elif task_type == "classification":
                label = "yes" if row_index % 2 else "no"
            else:
                label = f"class_{row_index % 3}"
            writer.writerow([row_index, row_index % 17, row_index // 5, label])  # noqa: FCN001 - row is positional-only
    else:
        raise ValueError(f"Unknown AutoML task type: {task_type}")

    return output.getvalue()
