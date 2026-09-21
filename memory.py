"""One retained-query budget per client: seeded replay plus any guard queries a method
consults online. A retained unit is one query id with all its relevance pairs."""
import numpy as np


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
    def used(self):
        return len(self._reserved) + len(self._replay)

    def reserve(self, qids):
        """Count guard queries consulted online against the budget; call before refill."""
        if self._replay:
            raise ValueError("reserve guard queries before refilling the replay memory")
        new = [q for q in qids if q not in self._reserved]
        if len(self._reserved) + len(new) > self.budget:
            raise ValueError(f"reserving {len(new)} more guard queries exceeds the "
                             f"budget of {self.budget}")
        self._reserved.extend(new)

    def refill(self, past):
        """Draw an equal share of each earlier experience's training queries, in experience
        order, seeded per experience. A small experience's unused slots pass to the larger
        ones; leftover slots from rounding go to the largest."""
        names = list(past)
        remaining = self.budget - len(self._reserved)
        chosen = {}
        for idx, name in enumerate(sorted(names, key=lambda n: (len(past[n]), n))):
            share = min(len(past[name]), remaining // (len(names) - idx))
            pool = sorted(past[name])
            rng = np.random.default_rng([self.seed, names.index(name)])
            picks = sorted(rng.choice(len(pool), size=share, replace=False).tolist())
            chosen[name] = [pool[i] for i in picks]
            remaining -= share
        self._replay = [q for name in names for q in chosen[name]]

    def record(self):
        return {"budget": self.budget, "seed": self.seed,
                "reserved": list(self._reserved), "replay": list(self._replay)}

    @classmethod
    def from_record(cls, record):
        return cls(record["budget"], record["seed"], record["reserved"], record["replay"])
