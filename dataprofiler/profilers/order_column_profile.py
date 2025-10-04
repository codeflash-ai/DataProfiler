"""Index profile analysis for individual col within structured profiling."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, Type, TypeVar, cast

import numpy as np
from pandas import DataFrame, Series

from . import profiler_utils
from .base_column_profilers import BaseColumnProfiler
from .profiler_options import OrderOptions


class Comparable(Protocol):
    """Protocol for ensuring comparable types, in this case both floats or strings."""

    @abstractmethod
    def __lt__(self: CT, other: CT) -> bool:
        """Protocol for ensuring comparable values."""
        pass


CT = TypeVar("CT", bound=Comparable)

# bc type in class attr causing issues, need to alias
AliasFloatType = Type[np.float64]
AliasStrType = Type[str]


class OrderColumn(BaseColumnProfiler["OrderColumn"]):
    """
    Index column profile subclass of BaseColumnProfiler.

    Represents a column in the dataset which is an index column.
    """

    type = "order"

    def __init__(self, name: str | None, options: OrderOptions = None) -> None:
        """
        Initialize column base properties and self.

        :param name: Name of the data
        :type name: String
        :param options: Options for the Order column
        :type options: OrderOptions
        """
        if options and not isinstance(options, OrderOptions):
            raise ValueError(
                "OrderColumn parameter 'options' must be of type" " OrderOptions."
            )
        self.order: str | None = None
        self._last_value: np.float64 | float | str | None = None
        self._first_value: np.float64 | float | str | None = None
        self._data_store_type: AliasStrType | AliasFloatType = np.float64
        self._piecewise: bool | None = False
        self.__calculations: dict = {}
        self._filter_properties_w_options(self.__calculations, options)
        super().__init__(name)

    @staticmethod
    def _is_intersecting(
        first_value1: CT,
        last_value1: CT,
        first_value2: CT,
        last_value2: CT,
    ) -> bool:
        """
        Check to see if the range of the datasets intersect.

        :param first_value1: beginning value of dataset 1
        :type first_value1: Float | String
        :param last_value1: last value of dataset 1
        :type last_value1: Float | String
        :param first_value2: beginning value of dataset 2
        :type first_value2: Float | String
        :param last_value2: last value of dataset 2
        :type last_value2: Float | String
        :return: Whether or not there is an intersection
        :rtype: Bool
        """
        # Some values may be in descending order so the true min/max must be
        # found
        low_value1 = min(first_value1, last_value1)
        high_value1 = max(first_value1, last_value1)
        low_value2 = min(first_value2, last_value2)
        high_value2 = max(first_value2, last_value2)
        is_intersecting = False

        if (
            (low_value2 < low_value1 < high_value2)
            or (low_value2 < high_value1 < high_value2)
            or (low_value1 < low_value2 < high_value1)
            or (low_value1 < high_value2 < high_value1)
            or (low_value1 == low_value2 and high_value1 == high_value2)
        ):
            is_intersecting = True
        return is_intersecting

    @staticmethod
    def _is_enveloping(
        first_value1: CT,
        last_value1: CT,
        first_value2: CT,
        last_value2: CT,
    ) -> bool:
        """
        Check to see if the range of the dataset 1 envelopes dataset 2.

        :param first_value1: beginning value of dataset 1
        :type first_value1: Float | String
        :param last_value1: last value of dataset 1
        :type last_value1: Float | String
        :param first_value2: beginning value of dataset 2
        :type first_value2: Float | String
        :param last_value2: last value of dataset 2
        :type last_value2: Float | String
        :return: Whether or not there is an intersection
        :rtype: Bool
        """
        # Some values may be in descending order so the true min/max must be
        # found
        low_value1 = min(first_value1, last_value1)
        high_value1 = max(first_value1, last_value1)
        low_value2 = min(first_value2, last_value2)
        high_value2 = max(first_value2, last_value2)
        is_enveloping = False
        if low_value1 < low_value2 and high_value1 > high_value2:
            is_enveloping = True
        return is_enveloping

    @BaseColumnProfiler._timeit(name="order")
    def _merge_order(
        self,
        order1: str,
        first_value1: CT,
        last_value1: CT,
        data_store_type1: AliasStrType | AliasFloatType,
        piecewise1: bool,
        order2: str,
        first_value2: CT,
        last_value2: CT,
        data_store_type2: AliasStrType | AliasFloatType,
        piecewise2: bool,
    ) -> tuple[str, CT | None, CT | None, bool, AliasStrType | AliasFloatType]:
        """
        Add the order of two datasets together.

        :param order1: order of original dataset
        :param first_value1: beginning value of original dataset
        :param last_value1: last value of original dataset
        :param data_store_type1: type of value for first_value1 and last_value1
        :param piecewise1: original dataset is piecewise or not
        :param order2: order of new dataset
        :param first_value2: beginning value of new dataset
        :param last_value2: last value of new dataset
        :param data_store_type2: type of value for first_value2 and last_value2
        :param piecewise2: new dataset is piecewise or not
        :type order1: String
        :type first_value1: Float | String
        :type last_value1: Float | String
        :type piecewise1: Boolean
        :type data_store_type1: Type[str] | Type[np.float64]
        :type order2: String
        :type first_value2: Float | String
        :type last_value2: Float | String
        :type data_store_type2: Type[str] | Type[np.float64]
        :type piecewise2: Boolean
        :return: order, first_value, last_value, piecewise, merged_data_store_type
        :rtype: String, Float | String, Float | String, Boolean, Type[str]
            | Type[np.float64]
        """
        # Return either order if one is None
        if not order1:
            return order2, first_value2, last_value2, piecewise2, data_store_type2
        elif not order2:
            return order1, first_value1, last_value1, piecewise1, data_store_type1

        merged_data_store_type: AliasStrType | AliasFloatType = np.float64
        if data_store_type1 is str or data_store_type2 is str:
            first_value1 = cast(CT, str(first_value1))
            last_value1 = cast(CT, str(last_value1))
            first_value2 = cast(CT, str(first_value2))
            last_value2 = cast(CT, str(last_value2))
            merged_data_store_type = str

        is_intersecting = self._is_intersecting(
            first_value1, last_value1, first_value2, last_value2
        )
        self_envelopes_other = self._is_enveloping(
            first_value1, last_value1, first_value2, last_value2
        )
        other_envelopes_self = self._is_enveloping(
            first_value2, last_value2, first_value1, last_value1
        )

        # Default initialization
        order = "random"
        first_value: CT | None = None
        last_value: CT | None = None

        if order1 == "random" or order2 == "random":
            order = "random"

        elif order1 == order2:
            if not is_intersecting or (piecewise1 and piecewise2):
                order = order1
            elif piecewise1 and self_envelopes_other:
                order = order1
            elif piecewise2 and other_envelopes_self:
                order = order1
            elif order1 == "constant value":
                order = order1
            else:
                order = "random"

        elif (order1 == "ascending" and order2 == "descending") or (
            order1 == "descending" and order2 == "ascending"
        ):
            order = "random"

        elif not is_intersecting:
            if order1 == "ascending" or order2 == "ascending":
                order = "ascending"
            else:
                order = "descending"

        else:
            if order1 == "constant value" and piecewise2:
                order = order2
            elif order2 == "constant value" and piecewise1:
                order = order1
            else:
                order = "random"

        # Set variables
        if order == "ascending":
            first_value = min(first_value1, first_value2)
            last_value = max(last_value1, last_value2)
        elif order == "descending":
            first_value = max(first_value1, first_value2)
            last_value = min(last_value1, last_value2)
        elif order == "random" or order == "constant value":
            first_value = min(first_value1, first_value2, last_value1, last_value2)
            last_value = max(first_value1, first_value2, last_value1, last_value2)

        piecewise = True
        if (
            order == "constant value" and first_value == last_value
        ) or order == "random":
            piecewise = False

        return order, first_value, last_value, piecewise, merged_data_store_type

    def __add__(self, other: OrderColumn) -> OrderColumn:
        """
        Merge the properties of two OrderColumn profiles.

        :param self: first profile
        :param other: second profile
        :type self: OrderColumn
        :type other: OrderColumn
        :return: Merged OrderColumn
        :rtype: OrderColumn
        """
        if not isinstance(other, OrderColumn):
            raise TypeError(
                "Unsupported operand type(s) for +: "
                "'OrderColumn' and '{}'".format(other.__class__.__name__)
            )

        merged_profile = OrderColumn(None)
        order, first_value, last_value, piecewise, data_store_type = self._merge_order(
            self.order,
            self._first_value,
            self._last_value,
            self._data_store_type,
            self._piecewise,
            other.order,
            other._first_value,
            other._last_value,
            other._data_store_type,
            other._piecewise,
        )

        merged_profile.order = order
        merged_profile._first_value = first_value
        merged_profile._last_value = last_value
        merged_profile._piecewise = piecewise
        merged_profile._data_store_type = data_store_type

        BaseColumnProfiler._add_helper(merged_profile, self, other)
        self._merge_calculations(
            merged_profile.__calculations, self.__calculations, other.__calculations
        )
        return merged_profile

    def report(self, remove_disabled_flag: bool = False) -> dict:
        """
        Private abstract method for returning report.

        :param remove_disabled_flag: flag to determine if disabled
            options should be excluded in the report.
        :type remove_disabled_flag: boolean
        """
        return self.profile

    @classmethod
    def load_from_dict(cls, data, config: dict | None = None):
        """
        Parse attribute from json dictionary into self.

        :param data: dictionary with attributes and values.
        :type data: dict[string, Any]
        :param config: options for loading column profiler params from dictionary
        :type config: Dict | None

        :return: Profiler with attributes populated.
        :rtype: CategoricalColumn
        """
        # This is an ambiguous call to super classes.
        data["_data_store_type"] = (
            str if data["_data_store_type"] == "str" else np.float64
        )
        profile = super().load_from_dict(data)
        try:
            if profile.sample_size and profile._data_store_type is np.float64:
                profile._first_value = np.float64(profile._first_value)
                profile._last_value = np.float64(profile._last_value)
        except ValueError:
            profile._first_value = data["_first_value"]
            profile._last_value = data["_last_value"]
        return profile

    @property
    def profile(self) -> dict:
        """
        Property for profile. Returns the profile of the column.

        :return:
        """
        return dict(order=self.order, times=self.times)

    def diff(self, other_profile: OrderColumn, options: dict = None) -> dict:
        """
        Generate the differences between the orders of two OrderColumns.

        :return: Dict containing the differences between orders in their
        appropriate output formats
        :rtype: dict
        """
        # Make sure other_profile's type matches this class
        super().diff(other_profile, options)

        differences = {
            "order": profiler_utils.find_diff_of_strings_and_bools(
                self.order, other_profile.order
            )
        }
        return differences

    @BaseColumnProfiler._timeit(name="order")
    def _get_data_order(
        self, df_series: Series, data_store_type: AliasStrType | AliasFloatType
    ) -> tuple[str, float, float, AliasStrType | AliasFloatType]:
        """
        Retrieve the order profile of a given data series.

        Return either: ascending, descending, constant value, or random.
        Additionally, return the first and last value of the series.

        :param df_series: a given column
        :type df_series: pandas.core.series.Series
        :param data_store_type: type of value for first_value and last_value
        :type data_store_type: Type[str] | Type[np.float64]
        :return: order, first_value, last_value, data_store_type
        :rtype: String, Float, Float, type, Type[str] | Type[np.float64]
        """
        # Fast path: try to avoid Python loop with NumPy for numeric/non-object
        # This covers most numeric, non-mixed columns, used heavily in tabular data
        try:
            if data_store_type is not str:
                # Try conversion up front, fall back if fails
                arr = df_series.values
                # If already float dtype, this is fast, else .astype() to safe float
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(float)
                else:
                    arr = arr  # no-op, keep as view
                data_store_type = np.float64
                _first = arr[0]
                _last = arr[0]
                n = arr.size
                # Vectorized check for constant
                all_equal = np.all(arr == arr[0])
                if all_equal:
                    return "constant value", _first, _last, data_store_type
                # Vectorized checks for monotonicity (strictly or non-strictly)
                asc = np.all(arr[:-1] <= arr[1:])
                desc = np.all(arr[:-1] >= arr[1:])
                if asc and not desc:
                    return "ascending", _first, arr[-1], data_store_type
                if desc and not asc:
                    return "descending", _first, arr[-1], data_store_type
                return "random", _first, arr[-1], data_store_type
        except Exception:
            data_store_type = str

        # Fallback for non-numeric (object/str or any conversion failure)
        arr = df_series.values
        _first = arr[0]
        _last = arr[0]

        n = arr.shape[0]
        # Optimize for all equal (constant value)
        constant = True
        for i in range(1, n):
            if arr[i] != _first:
                constant = False
                break
        if constant:
            return "constant value", _first, _last, data_store_type

        # Now try to determine order (ascending, descending, or random)
        order = None
        prev = _first
        for i in range(1, n):
            value = arr[i]
            if value < prev and order == "ascending":
                order = "random"
                break
            elif value < prev and order is None:
                order = "descending"
            elif value > prev and order == "descending":
                order = "random"
                break
            elif value > prev and order is None:
                order = "ascending"
            prev = value
        if not order:
            order = "constant value"
        return order, _first, arr[-1], data_store_type

    def _update_order(
        self,
        df_series: DataFrame,
        prev_dependent_properties: dict = None,
        subset_properties: dict = None,
    ) -> None:
        """
        Update order profile with order info attained from new dataset.

        Do this in following two steps:
        1. Get order information from input column data.
        2. Merge information between existing profile and new column
           order information.

        :param df_series: Data to be profiled
        :type df_series: pandas.DataFrame
        :param prev_dependent_properties: Contains all the previous properties
        that the calculations depend on.
        :type prev_dependent_properties: dict
        :param subset_properties: Contains the results of the properties of the
        subset before they are merged into the main data profile.
        :type subset_properties: dict
        :return: None
        """
        if self.order == "random":
            return
        order, first_value, last_value, data_store_type = self._get_data_order(
            df_series, self._data_store_type
        )

        (
            self.order,
            self._first_value,
            self._last_value,
            self._piecewise,
            self._data_store_type,
        ) = self._merge_order(
            self.order,
            self._first_value,
            self._last_value,
            self._data_store_type,
            self._piecewise,
            order,
            first_value,
            last_value,
            data_store_type,
            piecewise2=False,
        )

    def _update_helper(self, df_series_clean: Series, profile: dict) -> None:
        """
        Update col profile properties with clean dataset and its known null parameters.

        :param df_series_clean: df series with nulls removed
        :type df_series_clean: pandas.core.series.Series
        :param profile: ordered profile
        :type profile: dict
        :return: None
        """
        self._update_column_base_properties(profile)

    def update(self, df_series: Series) -> OrderColumn:
        """
        Update the column profile.

        :param df_series: df series
        :type df_series: pandas.core.series.Series
        :return: updated OrderColumn
        :rtype: OrderColumn
        """
        if len(df_series) == 0:
            return self

        profile = dict(sample_size=len(df_series))
        OrderColumn._update_order(self, df_series=df_series)
        BaseColumnProfiler._perform_property_calcs(
            self,
            self.__calculations,
            df_series=df_series,
            prev_dependent_properties={},
            subset_properties=profile,
        )
        self._update_helper(df_series, profile)

        return self
