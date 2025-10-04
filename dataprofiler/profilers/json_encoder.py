"""Contains ProfilerEncoder class."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..labelers.base_data_labeler import BaseDataLabeler
from . import (
    base_column_profilers,
    column_profile_compilers,
    numerical_column_stats,
    profile_builder,
    profiler_options,
)


class ProfileEncoder(json.JSONEncoder):
    """JSONify profiler objects and it subclasses and contents."""

    def default(self, to_serialize):
        """
        Specify how an object should be serialized.

        :param to_serialize: an object to be serialized
        :type to_serialize: a BaseColumnProfile object

        :raises: NotImplementedError

        :return: a datatype serializble by json.JSONEncoder
        """
        # Store frequently-accessed external classes as locals for faster isinstance checks
        pb = profile_builder
        bcp = base_column_profilers.BaseColumnProfiler
        nsm = numerical_column_stats.NumericStatsMixin
        cpc = column_profile_compilers.BaseCompiler
        po = profiler_options.BaseOption
        bp = pb.BaseProfiler
        scp = pb.StructuredColProfiler

        obj_type = type(to_serialize)

        # Fast type check for UnstructuredProfiler
        if isinstance(to_serialize, pb.UnstructuredProfiler):
            raise NotImplementedError(
                "UnstructuredProfiler serialization not supported."
            )

        # Group all profile-related type checks to a tuple so isinstance checks are slightly faster
        if isinstance(
            to_serialize,
            (bcp, nsm, cpc, po, bp, scp),
        ):
            # Avoid type(to_serialize) lookups twice
            return {"class": obj_type.__name__, "data": to_serialize.__dict__}

        # Type checks in order of most to least frequent, according to profile
        if isinstance(to_serialize, set):
            return list(to_serialize)

        if isinstance(to_serialize, np.integer):
            return int(to_serialize)

        if isinstance(to_serialize, np.ndarray):
            return to_serialize.tolist()

        # Use a tuple of types for Timestamp/Datetime, accessed via __class__ for possible speed
        if isinstance(to_serialize, (pd.Timestamp, datetime)):
            return to_serialize.isoformat()

        if isinstance(to_serialize, BaseDataLabeler):
            # TODO: This does not allow the user to serialize a model if it is loaded
            # "from_disk". Changes to BaseDataLabeler are needed for this feature
            if to_serialize._default_model_loc is None:
                raise ValueError(
                    "Serialization cannot be done on labelers with "
                    "_default_model_loc not set"
                )

            return {"from_library": to_serialize._default_model_loc}

        if callable(to_serialize):
            return to_serialize.__name__

        return super().default(to_serialize)
