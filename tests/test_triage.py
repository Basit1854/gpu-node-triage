#!/usr/bin/env python3
"""Tests for gpu-triage.py, driven by stub binaries on PATH.

There is no GPU on a laptop or a CI runner, so the only honest way to test the
parsing is to put a fake nvidia-smi and lspci ahead of the real ones on PATH
and shape their output the way the real tools shape it. The stubs are in
tests/fake-bin and are the interesting half of this file.

Run: python3 tests/test_triage.py
"""

import json
import os
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "gpu-triage.py"
FAKE_BIN = ROOT / "tests" / "fake-bin"


def sweep(scenario="healthy", bus_gpus=8, extra_args=(), env_extra=None,
          dmesg=None):
    """Run the script with the stubs on PATH. Returns (exit_code, findings)."""
    env = dict(os.environ)
    env["PATH"] = f"{FAKE_BIN}:{env.get('PATH', '')}"
    env["FAKE_SCENARIO"] = scenario
    env["FAKE_BUS_GPUS"] = str(bus_gpus)
    env["FAKE_DMESG"] = dmesg or ""
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--no-fabric", "--no-oob",
         "--timeout", "2", *extra_args],
        capture_output=True, text=True, env=env, timeout=120,
    )
    payload = json.loads(p.stdout)
    return p.returncode, payload["findings"]


def summaries(findings, layer=None, severity=None):
    return [f["summary"] for f in findings
            if (layer is None or f["layer"] == layer)
            and (severity is None or f["severity"] == severity)]


class TestHealthyNode(unittest.TestCase):
    def test_clean_node_reports_nothing_actionable(self):
        rc, findings = sweep("healthy")
        actionable = [f for f in findings
                      if f["severity"] in ("warn", "critical")]
        self.assertEqual(actionable, [], f"unexpected findings: {actionable}")

    def test_clean_node_sweep_is_complete(self):
        # The regression that prompted the blocked flag: a sweep that could not
        # run must never exit 0 and look like a clean node.
        rc, findings = sweep("healthy")
        blocked = [f for f in findings if f["blocked"]]
        self.assertEqual(blocked, [], f"unexpected blocked checks: {blocked}")
        self.assertEqual(rc, 0)


class TestKernelLog(unittest.TestCase):
    def test_hardware_xid_is_critical_and_quotes_the_line(self):
        rc, findings = sweep(dmesg="xid48-ecc.log")
        self.assertEqual(rc, 1)
        hits = [f for f in findings if f["summary"].startswith("Xid 48")]
        self.assertTrue(hits, summaries(findings, "kernel"))
        self.assertEqual(hits[0]["severity"], "critical")
        self.assertIn("double bit error", hits[0]["detail"])

    def test_repeated_xids_collapse_into_one_finding_with_a_count(self):
        # A flapping link produces hundreds of identical Xid 74s. One finding
        # that says how many beats hundreds of findings.
        rc, findings = sweep(dmesg="xid48-ecc.log")
        self.assertEqual(len([f for f in findings
                              if f["summary"].startswith("Xid 48")]), 1)
        detail = [f["detail"] for f in findings
                  if f["summary"].startswith("Xid 48")][0]
        self.assertIn("2 occurrence(s)", detail)

    def test_an_xid_line_is_not_also_counted_as_a_generic_nvrm_message(self):
        # The bare NVRM pattern matched every Xid line, so each one was
        # reported twice: once properly, once as an anonymous NVRM message.
        rc, findings = sweep(dmesg="xid48-ecc.log")
        self.assertEqual(
            [s for s in summaries(findings, "kernel") if "nvrm message" in s],
            [],
        )

    def test_workload_xid_is_a_warning_not_a_critical(self):
        # Xid 13 is a graphics exception, which is usually the customer's
        # kernel, not the board. Draining a node for it wastes capacity.
        rc, findings = sweep(dmesg="xid13-workload.log")
        self.assertEqual(summaries(findings, "kernel", "critical"), [])
        self.assertIn("Xid 13 present", summaries(findings, "kernel", "warn"))

    def test_the_driver_load_banner_is_not_a_finding(self):
        # Every healthy node logs "NVRM: loading NVIDIA UNIX x86_64 Kernel
        # Module" at boot. A pattern that flags the whole fleet gets ignored.
        rc, findings = sweep()
        self.assertEqual(summaries(findings, "kernel"), [])
        self.assertEqual(rc, 0)

    def test_fabric_errors_in_the_kernel_log_are_surfaced(self):
        rc, findings = sweep(dmesg="fabric-flap.log")
        self.assertEqual(rc, 1)
        self.assertTrue(
            any("mlx5 message" in s for s in summaries(findings, "kernel")),
            summaries(findings, "kernel"),
        )

    def test_pcie_errors_and_a_fallen_gpu_are_both_reported(self):
        # The AER corrected errors come first and the Xid 79 comes five minutes
        # later. Reading one without the other loses the story.
        rc, findings = sweep(dmesg="pcie-aer.log")
        labels = summaries(findings, "kernel")
        self.assertTrue(any("Xid 79" in s for s in labels), labels)
        self.assertTrue(any("pcie_aer message" in s for s in labels), labels)


class TestDevicePresence(unittest.TestCase):
    def test_missing_gpu_on_the_bus_is_critical(self):
        rc, findings = sweep("healthy", bus_gpus=7)
        self.assertEqual(rc, 1)
        self.assertTrue(
            any("7 GPUs on the PCIe bus" in s
                for s in summaries(findings, "presence", "critical")),
            summaries(findings, "presence"),
        )

    def test_expect_gpus_flag_is_respected(self):
        # A GB300 compute tray has 4, so 4 on the bus is correct there and a
        # fault on an HGX box. The default must not hard-code the node shape.
        rc, findings = sweep("healthy", bus_gpus=4, extra_args=("--expect-gpus", "4"))
        self.assertEqual(summaries(findings, "presence", "critical"), [])

    def test_driver_count_below_bus_count_is_critical(self):
        # 9 on the bus, 8 from the driver: the device is seated and the driver
        # is not binding it, which is a different fix from a dead board.
        rc, findings = sweep("healthy", bus_gpus=9,
                             extra_args=("--expect-gpus", "9"),
                             env_extra={"FAKE_DRIVER_GPUS": "8"})
        self.assertEqual(rc, 1)
        self.assertTrue(
            any("driver sees 8 GPUs, bus has 9" in s
                for s in summaries(findings, "presence", "critical")),
            summaries(findings, "presence"),
        )

    def test_hung_driver_is_a_finding_not_a_crash(self):
        rc, findings = sweep("hung")
        self.assertEqual(rc, 1)
        self.assertIn("nvidia-smi timed out",
                      summaries(findings, "presence", "critical"))

    def test_hung_driver_does_not_reprobe_in_later_steps(self):
        # Steps 3 and 4 used to call the same hung binary again, so one wedged
        # node cost three timeouts instead of one.
        rc, findings = sweep("hung")
        for layer in ("gpu_state", "nvlink"):
            skipped = [f for f in findings
                       if f["layer"] == layer and f["blocked"]]
            self.assertTrue(skipped, f"{layer} should record that it was skipped")


class TestEccAndRemapping(unittest.TestCase):
    def test_volatile_ecc_is_critical_and_names_the_gpu(self):
        rc, findings = sweep("ecc_volatile")
        self.assertEqual(rc, 1)
        self.assertTrue(
            any(s.startswith("GPU 2:") and "volatile" in s
                for s in summaries(findings, "gpu_state", "critical")),
            summaries(findings, "gpu_state"),
        )

    def test_aggregate_only_ecc_is_a_warning_not_a_critical(self):
        # Aggregate survives a driver reload and volatile does not, so errors
        # in aggregate alone are history, not something happening now.
        rc, findings = sweep("ecc_aggregate")
        self.assertEqual(summaries(findings, "gpu_state", "critical"), [])
        self.assertTrue(
            any("aggregate only" in s
                for s in summaries(findings, "gpu_state", "warn")),
            summaries(findings, "gpu_state"),
        )

    def test_volatile_errors_do_not_also_raise_the_aggregate_warning(self):
        # Both counters are non-zero in this scenario. One fault should produce
        # one finding, or the report pads itself.
        rc, findings = sweep("ecc_volatile")
        self.assertEqual(
            [s for s in summaries(findings, "gpu_state") if "aggregate only" in s],
            [],
        )

    def test_unsupported_ecc_counters_are_not_read_as_faults(self):
        # Unsupported counters come back "N/A", which is not a number above zero.
        rc, findings = sweep("no_ecc_support")
        self.assertEqual(summaries(findings, "gpu_state", "critical"), [])

    def test_remap_failure_is_critical(self):
        rc, findings = sweep("remap_failure")
        self.assertEqual(rc, 1)
        self.assertIn("GPU 3: row remapping failure",
                      summaries(findings, "gpu_state", "critical"))

    def test_remap_pending_is_a_warning(self):
        # Pending means it wants a reset. That is a maintenance window, not an
        # RMA, and conflating the two sends good boards back to the vendor.
        rc, findings = sweep("remap_pending")
        self.assertIn("GPU 5: row remap pending",
                      summaries(findings, "gpu_state", "warn"))
        self.assertEqual(summaries(findings, "gpu_state", "critical"), [])

    def test_unsupported_remap_fields_fall_back_instead_of_failing_step_3(self):
        # --query-gpu is all-or-nothing: on pre-Ampere the remapped_rows query
        # exits non-zero. ECC and thermals must still be reported, and the
        # remap check must fall back to parsing -q.
        rc, findings = sweep("pre_ampere")
        gpu_state = [f for f in findings if f["layer"] == "gpu_state"]
        self.assertFalse(
            any("could not query ECC" in f["summary"] for f in gpu_state),
            "the ECC query was collateral damage from the remap query",
        )
        # This scenario has an aggregate ECC error on GPU 1. It has to survive
        # the remap query failing beside it.
        self.assertTrue(
            any("GPU 1" in f["summary"] and "aggregate only" in f["summary"]
                for f in gpu_state),
            summaries(findings, "gpu_state"),
        )

    def test_fallback_parsing_catches_a_remap_failure(self):
        rc, findings = sweep("pre_ampere_fault")
        self.assertEqual(rc, 1)
        self.assertTrue(
            any("row remapping failure" in s
                for s in summaries(findings, "gpu_state", "critical")),
            summaries(findings, "gpu_state"),
        )


class TestThermalAndNvlink(unittest.TestCase):
    def test_hardware_slowdown_is_reported_with_the_temperature(self):
        rc, findings = sweep("thermal")
        hits = [f for f in findings
                if f["layer"] == "gpu_state" and "slowdown" in f["summary"]]
        self.assertTrue(hits, summaries(findings, "gpu_state"))
        self.assertIn("91", hits[0]["detail"])

    def test_both_slowdown_reasons_are_distinguished(self):
        # Hardware slowdown and software thermal slowdown have different causes
        # and the second one used to be queried and then never read.
        rc, findings = sweep("thermal")
        labels = summaries(findings, "gpu_state", "warn")
        self.assertTrue(any("hardware slowdown" in s for s in labels), labels)
        self.assertTrue(any("software thermal slowdown" in s for s in labels), labels)

    def test_inactive_nvlink_is_critical(self):
        rc, findings = sweep("nvlink_down")
        self.assertEqual(rc, 1)
        self.assertTrue(
            any("NVLink(s) inactive" in s
                for s in summaries(findings, "nvlink", "critical")),
            summaries(findings, "nvlink"),
        )


class TestExitCodes(unittest.TestCase):
    def test_incomplete_sweep_exits_2(self):
        # No stubs on PATH, so nothing can run.
        p = subprocess.run(
            [sys.executable, str(SCRIPT), "--json"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PATH": "/nonexistent"},
        )
        self.assertEqual(p.returncode, 2)
        self.assertFalse(json.loads(p.stdout)["complete"])

    def test_a_real_finding_outranks_an_incomplete_sweep(self):
        # Fabric and OOB tooling is absent here, so the sweep is incomplete,
        # but an uncorrectable ECC error is the thing a human needs to see.
        env = dict(os.environ)
        env["PATH"] = f"{FAKE_BIN}:{env.get('PATH', '')}"
        env["FAKE_SCENARIO"] = "ecc_volatile"
        p = subprocess.run(
            [sys.executable, str(SCRIPT), "--json"],
            capture_output=True, text=True, env=env, timeout=120,
        )
        payload = json.loads(p.stdout)
        self.assertFalse(payload["complete"])
        self.assertEqual(p.returncode, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
