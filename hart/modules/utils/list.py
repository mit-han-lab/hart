from typing import Optional, Union

__all__ = [
    "list_sum",
    "list_mean",
    "weighted_list_sum",
    "list_join",
    "val2list",
    "val2tuple",
    "squeeze_list",
]


def list_sum(x: list) -> any:
    """Return the sum of a list of objects.

    can be int, float, torch.Tensor, np.ndarray, etc.
    can be used for adding losses
    """
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> any:
    """Return the mean of a list of objects (can be int, float, torch.Tensor, np.ndarray, etc)."""
    return list_sum(x) / len(x)


def weighted_list_sum(x: list, weights: list) -> any:
    """Return a weighted sum of the list (can be used for adding loss)."""
    assert len(x) == len(weights)
    return (
        x[0] * weights[0]
        if len(x) == 1
        else x[0] * weights[0] + weighted_list_sum(x[1:], weights[1:])
    )


def list_join(x: list, sep="\t", format_str="%s") -> str:
    """Convert a list of objects to string based on the given format.

    (usually used for getting logs during training).
    """
    return sep.join([format_str % val for val in x])


def val2list(x: Union[list, tuple, any], repeat_time=1) -> list:
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(
    x: Union[list, tuple, any], min_len: int = 1, idx_repeat: int = -1
) -> tuple:
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def squeeze_list(x: Optional[list]) -> Union[list, any]:
    """Return the first item of the given list if the list only contains one item.

    usually used in args parsing
    """
    if x is not None and len(x) == 1:
        return x[0]
    else:
        return x
