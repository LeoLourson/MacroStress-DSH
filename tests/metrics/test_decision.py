"""The four-step decision rule of section 5."""

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
