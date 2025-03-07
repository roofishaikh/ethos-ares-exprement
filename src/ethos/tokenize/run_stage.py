import functools
import random
import time
from pathlib import Path

import polars as pl
from loguru import logger
from MEDS_transforms.mapreduce.utils import rwlock_wrap

from ..vocabulary import Vocabulary


def run_stage(
    in_fps,
    out_fps,
    *transform_fns,
    params={},
    vocab=None,
    agg_to=None,
    agg_params=None,
    worker=1,
    **kwargs,
):
    """This function can be run in parallel by multiple workers."""

    if vocab is not None:
        params = {"vocab": Vocabulary.from_path(vocab), **params}

    transforms_to_run = [
        functools.partial(transform_fn, **params) for transform_fn in transform_fns
    ]

    fps = list(zip(in_fps, out_fps))
    random.shuffle(fps)

    for in_fp, out_fp in fps:
        rwlock_wrap(
            in_fp,
            out_fp,
            functools.partial(pl.read_parquet, use_pyarrow=True),
            lambda df, out_: df.write_parquet(out_, use_pyarrow=True),
            compute_fn=lambda df: functools.reduce(lambda df, fn: fn(df), transforms_to_run, df),
        )

    if agg_to is not None:
        if worker == 1:
            i = 0
            while not all(out_fp.exists() for out_fp in out_fps):
                time.sleep(1)
                if i > 5:
                    logger.warning(
                        f"Waiting for {[out_fp.relative_to(in_fps[0]) for out_fp in out_fps]}"
                    )
                i += 1

            agg_params = agg_params or {}
            transform_fns[-1].agg(in_fps=out_fps, out_fp=agg_to, **agg_params)
        else:
            while not Path(agg_to).exists():
                time.sleep(1)
