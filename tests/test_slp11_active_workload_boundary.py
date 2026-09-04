"""Guard the active OMF workload surface from the historical dense prototype."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "workloads"
HISTORICAL_DENSE_MODULE = "modules/slp-1-1-world/module.yaml"


class ActiveWorkloadBoundaryTest(unittest.TestCase):
    def test_no_active_workload_references_historical_dense_world_module(self) -> None:
        offenders = []
        for path in sorted(WORKLOADS.glob("*.yaml*")):
            if HISTORICAL_DENSE_MODULE in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            "historical dense SLp-1.1 module re-entered the active workload surface",
        )

    def test_obsolete_dense_workload_names_remain_retired(self) -> None:
        for name in ("slp-1-1-pretrain.yaml", "slp-1-1-world-smoke.yaml"):
            with self.subTest(name=name):
                self.assertFalse((WORKLOADS / name).exists())


if __name__ == "__main__":
    unittest.main()
