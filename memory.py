"""One retained-query budget per client: seeded replay plus any guard queries a method
consults online. A retained unit is one query id with all its relevance pairs."""
import numpy as np

GUARD_STREAM = 1


def equal_share(past, slots, seed):
    """An equal share of ``slots`` from each list in ``past``, in the order of ``past`` and
    seeded per list. A short list's unused slots pass to the longer ones; leftover slots
    from rounding go to the longest."""
    names = list(past)
    chosen = {}
    for idx, name in enumerate(sorted(names, key=lambda n: (len(past[n]), n))):
        share = min(len(past[name]), slots // (len(names) - idx))
        pool = sorted(past[name])
        rng = np.random.default_rng([*seed, names.index(name)])
        picks = sorted(rng.choice(len(pool), size=share, replace=False).tolist())
        chosen[name] = [pool[i] for i in picks]
        slots -= share
    return [q for name in names for q in chosen[name]]


class ReplayMemory:
    def __init__(self, budget, seed, reserved=(), replay=()):
        self.budget = int(budget)
        self.seed = int(seed)
        self._reserved = list(reserved)
        self._replay = list(replay)

    @property
    def ids(self):
        return list(self._replay)

    @property
    def reserved(self):
        return list(self._reserved)

    @property
    def used(self):
        return len(self._reserved) + len(self._replay)

    def reserve(self, qids):
        """Count guard queries consulted online against the budget."""
        new = [q for q in qids if q not in self._reserved]
        if self.used + len(new) > self.budget:
            raise ValueError(f"reserving {len(new)} more guard queries exceeds the "
                             f"budget of {self.budget}")
        self._reserved.extend(new)

    def redraw_guard(self, past, size):
        """Replace the guard queries with an equal share of ``size`` from each earlier
        experience's guard split, drawn apart from replay; refill replay afterwards."""
        if size > self.budget:
            raise ValueError(f"{size} guard queries exceed the budget of {self.budget}")
        self._reserved = equal_share(past, size, [self.seed, GUARD_STREAM])

    def refill(self, past):
        """Fill the slots the guard queries leave with an equal share of each earlier
        experience's training queries, in experience order, seeded per experience."""
        self._replay = equal_share(past, self.budget - len(self._reserved), [self.seed])

    def record(self):
        return {"budget": self.budget, "seed": self.seed,
                "reserved": list(self._reserved), "replay": list(self._replay)}

    @classmethod
    def from_record(cls, record):
        return cls(record["budget"], record["seed"], record["reserved"], record["replay"])
