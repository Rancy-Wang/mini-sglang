import torch

from minisgl.scheduler.drop_recovery import plan_recovery


def test_recursive_dependencies_reuse_resident_suffix():
    plan = plan_recovery(
        torch.tensor([True, False, True, False, True, True]),
        torch.tensor([10, 4, 10, 10, 10, 10, 10, 10]), 8,
    )
    assert plan.intervals == ((1, 2), (3, 4), (6, 8))
    assert plan.required_prefix.tolist() == [True] * 6
    assert plan.next_interval(2) == (3, 4)


def test_dropped_holes_do_not_trigger_repair():
    plan = plan_recovery(torch.tensor([True, False, False, True]),
                         torch.tensor([10, 4, 4, 10, 10]), 5)
    assert plan.intervals == ((4, 5),)
    assert plan.required_prefix.tolist() == [True, False, False, True]


def test_dependency_drop_at_query_boundary_is_invisible():
    plan = plan_recovery(torch.tensor([False, True, False]),
                         torch.tensor([2, 10, 10, 10]), 4)
    assert plan.intervals == ((2, 4),)
    assert plan.required_prefix.tolist() == [False, True, True]
