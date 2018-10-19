import numpy as np


class irregular_array(list):
    """irregular array---subclass of list with depth and nonnumeric filters

    # Notes:
    This is a very basic type, with very limited use case. It is used only for
    calibration of cameras, where we often have missing values. It is common to
    use None in place of missing data when returning calculated properties, and
    so we have this class to handle the "lists-of-lists-and-None" problem i.e.
    an irregular array.

    Good performance is not garranteed whatsoever. Use with care!!
    """
    def __init__(self, data):
        super().__init__(data)

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        try:
            return CustomList(result)
        except TypeError:
            return result

    _seqcls = (list, tuple, np.ndarray)

    @property
    def shape(self):
        return _find_shape(list(self), self._seqcls)

    def filter_none(self, masks=None):
        if masks is None:
            masks = (self,)
        return _filter_none(list(self), masks, self._seqcls)


def _find_shape(element, seqence_classes):
    shape = tuple()
    if not any(isinstance(element, c) for c in seqence_classes):
        return shape
    for subelement in element:
        s = _find_shape(subelement, seqence_classes)
        if s != tuple() and (shape == tuple() or s[0] > shape[0]):
            shape = s
    return (len(element),) + shape


def _filter_none(element, masks, seqence_classes):
    if not any(isinstance(element, c) for c in seqence_classes):
        return element
    if all(isinstance(mask, np.ndarray) and mask.dtype != object
            for mask in masks):  # cannot contain any more Nones
        return element
    return [_filter_none(arr, mas, seqence_classes)
            for *mas, arr in zip(*masks, element)
            if all(ma is not None for ma in mas)]
