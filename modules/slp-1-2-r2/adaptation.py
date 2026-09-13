"""Human SL adaptation with optional permitted human quantitative replay."""

import hashlib
import json
import random


class Sampler:
    def __init__(self, labels, replay, *, sl_fraction=0.75, seed=731):
        if not 0 < sl_fraction <= 1 or (replay is None and sl_fraction != 1):
            raise ValueError("Invalid SL/replay allocation")
        self.labels, self.replay, self.fraction = labels, replay, sl_fraction
        self.random = random.Random(seed)
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    "labels": labels.signature,
                    "replay": getattr(replay, "signature", None),
                    "sl_fraction": sl_fraction,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def draw(self, max_queries=128):
        if self.random.random() < self.fraction:
            return ("sl", self.labels.draw(1))
        return ("quantitative", self.replay.draw(max_queries))

    def state_dict(self):
        return {
            "signature": self.signature,
            "random": self.random.getstate(),
            "labels": self.labels.state_dict(),
            "replay": self.replay.state_dict() if self.replay else None,
        }

    def load_state_dict(self, state):
        if state["signature"] != self.signature:
            raise ValueError("Changed adaptation mixture")
        self.random.setstate(state["random"])
        self.labels.load_state_dict(state["labels"])
        if self.replay:
            self.replay.load_state_dict(state["replay"])


class Builder:
    def __init__(self, labels, replay):
        self.builders = {"sl": labels, "quantitative": replay}

    def __call__(self, units):
        groups = {}
        for kind, unit in units:
            groups.setdefault(kind, []).append(unit)
        batches = [self.builders[kind](values) for kind, values in groups.items()]
        keys = set(batches[0])
        if any(set(b) != keys for b in batches):
            raise ValueError("Adaptation batch contracts differ")
        result = {}
        for key in keys:
            values = [b[key] for b in batches]
            if any(
                v.ndim != values[0].ndim or v.dtype != values[0].dtype for v in values
            ):
                raise ValueError("Adaptation tensor contracts differ")
            shape = (sum(v.shape[0] for v in values),) + tuple(
                max(v.shape[d] for v in values) for d in range(1, values[0].ndim)
            )
            output = values[0].new_zeros(shape)
            at = 0
            for value in values:
                output[
                    (slice(at, at + len(value)),)
                    + tuple(slice(0, n) for n in value.shape[1:])
                ] = value
                at += len(value)
            result[key] = output
        return result
