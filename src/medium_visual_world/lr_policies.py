"""LR policies for Medium World with genuine task-distribution boundaries.

No model or optimizer state is recreated at a phase transition. Cosine learning
rates are explicit per optimizer update; plateau checks are 100 steps apart
within their respective global or phase-local clock.
"""
import math
from bisect import bisect_right
import torch
from src.medium_visual_world.curriculum import TASKS

POLICIES = (
    "constant", "global_cosine", "reset_cosine",
    "global_plateau", "reset_plateau",
)


def task_mix(phase):
    n = phase["presentations"]
    return tuple(phase["by_template"].get(task, 0) / n for task in TASKS)


def effective_phases(phases):
    """Merge adjacent blocks with identical task-mixture proportions."""
    merged = []
    for phase in phases:
        name = phase["name"]
        if merged and merged[-1]["mix"] == task_mix(phase):
            merged[-1]["end_step"] = phase["end_step"]
            merged[-1]["names"].append(name)
        else:
            merged.append({"start_step": phase["start_step"],
                           "end_step": phase["end_step"],
                           "names": [name], "mix": task_mix(phase)})
    return merged


class LRPolicy:
    def __init__(self, name, optimizer, segments, initial_lr=1e-4,
                 final_mixed_lr=5e-5, min_lr=1e-6, factor=2/3,
                 patience=2, threshold=0.001, check_every=100):
        self.name = name
        self.optimizer = optimizer
        self.segments = segments
        self.total_steps = segments[-1]["end_step"]
        self.starts = [part["start_step"] for part in segments]
        self.initial_lr = initial_lr
        self.final_mixed_lr = final_mixed_lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.check_every = check_every
        self.scheduler = None
        self.reset_events = []
        self._set_lr(initial_lr)

    def _set_lr(self, value):
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    def _part_at(self, update_step):
        return bisect_right(self.starts, update_step) - 1

    def _restart_lr(self, part_index):
        part = self.segments[part_index]
        if part_index > 0 and "common_mixed" in part["names"]:
            return self.final_mixed_lr
        return self.initial_lr

    def _cosine(self, step, start, end, peak):
        progress = (step - start) / max(1, end - start - 1)
        return self.min_lr + (peak - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2

    def preview(self, step):
        if step >= self.total_steps:
            return self.optimizer.param_groups[0]["lr"]
        if self.name == "global_cosine":
            return self._cosine(step, 0, self.total_steps, self.initial_lr)
        if self.name == "reset_cosine":
            index = self._part_at(step)
            part = self.segments[index]
            return self._cosine(step, part["start_step"], part["end_step"],
                                self._restart_lr(index))
        return self.optimizer.param_groups[0]["lr"]

    def before_update(self, step):
        lr = self.preview(step)
        self._set_lr(lr)
        return lr

    def _new_plateau(self, loss):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor,
            patience=self.patience, threshold=self.threshold,
            threshold_mode="abs", min_lr=self.min_lr)
        self.scheduler.step(float(loss))

    def observe(self, step, dev_loss):
        """Call after evaluating the model at step, including step 0.

        Global plateau checks occur at steps 100, 200, ... . Phase plateau
        checks occur every 100 updates from the most recent real boundary.
        Reset histories and LR only when the task distribution changes.
        """
        if step == 0:
            if self.name.endswith("plateau"):
                self._new_plateau(dev_loss)
            return

        old = self._part_at(step - 1)
        part = self.segments[old]
        if self.name == "global_plateau" and step % self.check_every == 0:
            self.scheduler.step(float(dev_loss))
        elif self.name == "reset_plateau":
            if (step - part["start_step"]) % self.check_every == 0:
                self.scheduler.step(float(dev_loss))

        next_index = old + 1
        is_transition = (next_index < len(self.segments) and
                         step == self.segments[next_index]["start_step"])
        if is_transition and self.name.startswith("reset_"):
            before = self.optimizer.param_groups[0]["lr"]
            peak = self._restart_lr(next_index)
            if self.name == "reset_plateau":
                self._set_lr(peak)
                self._new_plateau(dev_loss)
            self.reset_events.append({"step": step, "before_lr": before,
                                      "restart_lr": peak,
                                      "to": self.segments[next_index]["names"]})


def evaluation_steps(phases, segments, check_every=100):
    total = segments[-1]["end_step"]
    steps = set(range(0, total + 1, check_every))
    for phase in phases[:-1]:
        boundary = phase["end_step"]
        steps.add(boundary)
        steps.add(min(total, boundary + check_every))
    for part in segments:
        steps.update(range(part["start_step"] + check_every,
                           part["end_step"], check_every))
    return sorted(steps)
