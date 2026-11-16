"""The four-step decision rule."""

from mst.metrics.decision import decide


def test_step_one_rejects_a_candidate_that_failed_admission():
    verdict = decide(
        candidate_crps=0.10,
        benchmark_crps=0.50,
        incumbent_crps=0.50,
        se_vs_benchmark=0.05,
        se_vs_incumbent=0.05,
        candidate_admission_shares={"band_outside_domain": 0.06, "band_wider_than_history": 0.01},
        incumbent_admission_shares={"band_outside_domain": 0.01, "band_wider_than_history": 0.01},
    )
    assert verdict["verdict"] == "rejected_admission" and verdict["step"] == 1


def test_step_two_accepts_a_two_se_improvement_with_unchanged_admission():
    verdict = decide(
        candidate_crps=0.10,
        benchmark_crps=0.50,
        incumbent_crps=0.50,
        se_vs_benchmark=0.05,
        se_vs_incumbent=0.05,
        candidate_admission_shares={"band_outside_domain": 0.01, "band_wider_than_history": 0.01},
        incumbent_admission_shares={"band_outside_domain": 0.01, "band_wider_than_history": 0.02},
    )
    assert verdict["verdict"] == "accepted" and verdict["step"] == 2


def test_step_two_needs_both_benchmarks_for_the_first_version_only_the_benchmark():
    without = decide(
        candidate_crps=0.10,
        benchmark_crps=0.50,
        incumbent_crps=None,
        se_vs_benchmark=0.05,
        se_vs_incumbent=None,
        candidate_admission_shares={"band_outside_domain": 0.01},
        incumbent_admission_shares=None,
    )
    assert without["verdict"] == "accepted"
    # With an incumbent, an improvement only over the benchmark is not enough.
    guarded = decide(
        candidate_crps=0.10,
        benchmark_crps=0.50,
        incumbent_crps=0.12,
        se_vs_benchmark=0.05,
        se_vs_incumbent=0.05,
        candidate_admission_shares={"band_outside_domain": 0.01},
        incumbent_admission_shares={"band_outside_domain": 0.01},
    )
    assert guarded["verdict"] == "inconclusive"


def test_step_two_rejects_when_an_admission_share_grew():
    verdict = decide(
        candidate_crps=0.10,
        benchmark_crps=0.50,
        incumbent_crps=0.50,
        se_vs_benchmark=0.05,
        se_vs_incumbent=0.05,
        candidate_admission_shares={"band_outside_domain": 0.02, "band_wider_than_history": 0.01},
        incumbent_admission_shares={"band_outside_domain": 0.01, "band_wider_than_history": 0.01},
    )
    assert verdict["verdict"] == "inconclusive" and verdict["step"] == 4
    assert verdict["admission_grown"] == ["band_outside_domain"]


def test_step_three_rejects_a_two_se_regression():
    worse = decide(
        candidate_crps=0.70,
        benchmark_crps=0.50,
        incumbent_crps=0.50,
        se_vs_benchmark=0.05,
        se_vs_incumbent=0.05,
        candidate_admission_shares={"band_outside_domain": 0.01},
        incumbent_admission_shares={"band_outside_domain": 0.01},
    )
    assert worse["verdict"] == "rejected_worse" and worse["step"] == 3


def test_step_four_stays_inconclusive_within_two_se():
    verdict = decide(
        candidate_crps=0.51,
        benchmark_crps=0.50,
        incumbent_crps=0.50,
        se_vs_benchmark=0.05,
        se_vs_incumbent=0.05,
        candidate_admission_shares={"band_outside_domain": 0.01},
        incumbent_admission_shares={"band_outside_domain": 0.01},
    )
    assert verdict["verdict"] == "inconclusive" and verdict["step"] == 4


def test_the_same_inputs_give_the_same_verdict():
    kwargs = dict(
        candidate_crps=0.10,
        benchmark_crps=0.50,
        incumbent_crps=0.50,
        se_vs_benchmark=0.05,
        se_vs_incumbent=0.05,
        candidate_admission_shares={"band_outside_domain": 0.01},
        incumbent_admission_shares={"band_outside_domain": 0.01},
    )
    assert decide(**kwargs) == decide(**kwargs)


def _rows(unit_count, scale):
    return [
        {
            "episode": f"E{u}",
            "month": t,
            "quantity": q,
            "model": scale * (1 + u + t),
            "walk": 1 + u + t,
        }
        for u in range(unit_count)
        for t in (1, 2)
        for q in ("a", "b")
    ]


def test_on_rows_reads_the_ratio_against_the_benchmark_and_the_incumbent():
    from mst.metrics.decision import on_rows

    shares = {"band_outside_domain": 0.0, "band_wider_than_history": 0.0}
    verdict = on_rows(
        _rows(6, 0.5),
        _rows(6, 1.0),
        unit="episode",
        benchmark="walk",
        block=1,
        seed=1,
        admission=shares,
        incumbent_admission=shares,
    )
    assert verdict["vs_benchmark"]["point"] == 0.5
    assert verdict["vs_incumbent"]["point"] == 0.5
    assert verdict["verdict"] == "accepted"


def test_on_rows_refuses_two_reports_on_different_cells():
    import pytest

    from mst.metrics.decision import on_rows

    with pytest.raises(ValueError, match="do not share 4 cells"):
        on_rows(
            _rows(2, 0.5),
            _rows(1, 1.0),
            unit="episode",
            benchmark="walk",
            block=1,
            seed=1,
            admission={},
            incumbent_admission={},
        )


def test_on_rows_with_one_unit_detects_nothing():
    from mst.metrics.decision import on_rows

    verdict = on_rows(
        _rows(1, 0.5),
        _rows(1, 1.0),
        unit="episode",
        benchmark="walk",
        block=1,
        seed=1,
        admission={},
        incumbent_admission={},
    )
    assert verdict["verdict"] == "inconclusive"
