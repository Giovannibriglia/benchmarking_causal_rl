"""feat/table-formatting-enhancements — table formatting unit tests.

Tests:
- best_indices_per_column with max, min, ties, NaN.
- format_cell with bold and precision.
- metric_direction lookups (incl. base-name _mean fallback and default).
- detect_sweep_families across various YAML directory layouts.
- family_label generates expected labels for known families.
- strength_to_float_label decimal rendering.
"""

import numpy as np
import pytest
from src.benchmarking.table_formatting import (
    best_indices_per_column,
    detect_sweep_families,
    family_label,
    format_cell,
    metric_direction,
    strength_to_float_label,
)


class TestBestIndicesPerColumn:
    def test_max_simple(self):
        v = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        best = best_indices_per_column(v, "max")
        assert best == [{2}, {2}]  # row 2 is best in both columns

    def test_min_simple(self):
        v = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        best = best_indices_per_column(v, "min")
        assert best == [{0}, {0}]

    def test_tied_bests_bolded_together(self):
        v = np.array([[1.0, 4.0], [3.0, 4.0], [3.0, 5.0]])
        best = best_indices_per_column(v, "max")
        assert best == [{1, 2}, {2}]

    def test_nan_excluded(self):
        v = np.array([[1.0, 4.0], [np.nan, 5.0], [3.0, 6.0]])
        best = best_indices_per_column(v, "max")
        assert best == [{2}, {2}]

    def test_all_nan_column_empty(self):
        v = np.array([[np.nan, 4.0], [np.nan, 5.0]])
        best = best_indices_per_column(v, "max")
        assert best == [set(), {1}]


class TestFormatCell:
    def test_no_bold(self):
        assert (
            format_cell(123.456, 12.345, is_best=False, precision=1) == "123.5 ± 12.3"
        )

    def test_with_bold(self):
        assert (
            format_cell(123.456, 12.345, is_best=True, precision=1)
            == "\\textbf{123.5 ± 12.3}"
        )

    def test_nan_returns_dash(self):
        assert format_cell(np.nan, np.nan, is_best=False, precision=1) == "—"

    def test_precision_override(self):
        assert (
            format_cell(1.2345, 0.6789, is_best=False, precision=3) == "1.234 ± 0.679"
        )


class TestMetricDirection:
    def test_known_max(self):
        assert metric_direction("eval_return_mean") == "max"

    def test_known_min(self):
        assert metric_direction("td_error") == "min"

    def test_base_name_mean_fallback(self):
        # Logical metric name resolves via the <name>_mean key.
        assert metric_direction("eval_return") == "max"

    def test_unknown_defaults_max(self):
        assert metric_direction("some_new_metric") == "max"


class TestDetectSweepFamilies:
    def test_basic_curiosity_family(self, tmp_path):
        for s in ["000", "025", "050", "075", "100"]:
            (tmp_path / f"online_discrete_curiosity_{s}.yaml").touch()
        families = detect_sweep_families(tmp_path)
        assert "online_discrete_curiosity" in families
        assert len(families["online_discrete_curiosity"]) == 5

    def test_singleton_excluded(self, tmp_path):
        (tmp_path / "online_discrete.yaml").touch()
        (tmp_path / "online_discrete_curiosity_000.yaml").touch()
        families = detect_sweep_families(tmp_path)
        # online_discrete is a singleton (no strength suffix); curiosity has only
        # one member (< 3) so neither is reported as a family.
        assert "online_discrete_curiosity" not in families
        assert "online_discrete" not in families

    def test_sigma_sweep_in_middle(self, tmp_path):
        for s in ["000", "025", "050", "075", "100"]:
            (tmp_path / f"confounded_sigma_{s}_discrete_gated.yaml").touch()
        families = detect_sweep_families(tmp_path)
        assert "confounded_sigma_discrete_gated" in families
        assert len(families["confounded_sigma_discrete_gated"]) == 5

    def test_mid_string_tail_does_not_collide(self, tmp_path):
        # Two families share the _<digits>_ prefix but differ in tail; they must
        # not be merged into one family.
        for s in ["000", "025", "050"]:
            (tmp_path / f"confounded_sigma_{s}_discrete.yaml").touch()
            (tmp_path / f"confounded_sigma_{s}_discrete_gated.yaml").touch()
        families = detect_sweep_families(tmp_path)
        assert "confounded_sigma_discrete" in families
        assert "confounded_sigma_discrete_gated" in families
        assert len(families["confounded_sigma_discrete"]) == 3
        assert len(families["confounded_sigma_discrete_gated"]) == 3

    def test_members_sorted_by_strength(self, tmp_path):
        for s in ["100", "000", "050"]:
            (tmp_path / f"online_discrete_curiosity_{s}.yaml").touch()
        families = detect_sweep_families(tmp_path)
        strengths = [s for s, _ in families["online_discrete_curiosity"]]
        assert strengths == ["000", "050", "100"]


class TestFamilyLabel:
    @pytest.mark.parametrize(
        "stem,expected_contains",
        [
            ("online_discrete_curiosity", "Curiosity"),
            ("online_continuous_anti_reward", "Anti reward"),
            ("confounded_sigma_discrete_gated", "Confounded sigma"),
            ("online_confounded_sigma_discrete_gated", "Online confounded sigma"),
        ],
    )
    def test_label_strips_modifiers(self, stem, expected_contains):
        label = family_label(stem)
        assert expected_contains in label or expected_contains.lower() in label.lower()

    def test_online_dropped_for_plain_dial(self):
        # online pairs with a regime descriptor -> dropped.
        assert family_label("online_discrete_curiosity") == "Curiosity"

    def test_online_kept_for_confounded(self):
        # online distinguishes the online confounded variant from its offline
        # sibling -> kept.
        assert family_label("online_confounded_sigma_discrete_gated") == (
            "Online confounded sigma"
        )

    def test_masked_becomes_suffix(self):
        assert family_label("online_masked_discrete_curiosity") == "Curiosity masked"


class TestStrengthToFloatLabel:
    @pytest.mark.parametrize(
        "code,expected",
        [("000", "0.00"), ("025", "0.25"), ("050", "0.50"), ("100", "1.00")],
    )
    def test_decimal_rendering(self, code, expected):
        assert strength_to_float_label(code) == expected
