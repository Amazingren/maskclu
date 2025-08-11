class AverageMeter:
    def __init__(self, items=None):
        # Initialize with item names or None for a single
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        # Reset values, sums, and counts for all items
        self._val = [0.0] * self.n_items
        self._sum = [0.0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        # Update can handle a single value or a list/tuple of values
        values = [values] if not isinstance(values, (list, tuple)) else values
        if len(values) != self.n_items:
            raise ValueError("Number of values provided does not match number of tracked items")
        
        for idx, value in enumerate(values):
            self._val[idx] = value
            self._sum[idx] += value
            self._count[idx] += 1

    def val(self, idx=None):
        # Get current values; if idx is None, return list of values for all items
        return self._val if idx is None else self._val[idx]

    def avg(self, idx=None):
        # Calculate and return the average; handle division by zero
        if idx is None:
            return [s / c if c != 0 else 0 for s, c in zip(self._sum, self._count)]
        else:
            return self._sum[idx] / self._count[idx] if self._count[idx] != 0 else 0
        
        
class AccMetric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc'] # type: ignore
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc # type: ignore
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict
