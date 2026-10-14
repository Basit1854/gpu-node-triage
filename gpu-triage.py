#!/usr/bin/env python3
"""
Read-only triage sweep for a suspect GPU node.

Runs steps 1-6 of the ordering documented in README.md. Deliberately does not
cordon, reset, reboot, or run dcgmi diag: two of those destroy the evidence you
are trying to collect, and all of them are a human's decision.

Exit codes:
    0  nothing found
    1  findings that need a human
    2  sweep could not complete (missing tooling, insufficient privileges)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys

# Kernel-log patterns worth surfacing. Deliberately narrow: a grep for "error"
# on a busy node returns pages of noise that nobody reads.
KERNEL_PATTERNS = {
    "xid": r"NVRM: Xid \((?:[^)]*)\): (\d+)",
    # Not a bare "NVRM:". Every healthy node logs "NVRM: loading NVIDIA UNIX
    # x86_64 Kernel Module" at boot, so the bare prefix flags the entire fleet
    # and the output stops being read.
    "nvrm": r"NVRM:.*(?:error|fail|timed out|timeout|unable|cannot|invalid|"
            r"lost|reset)",
    "mlx5": r"mlx5_core.*(?:error|down|fail)",
    "pcie_aer": r"AER:.*(?:error|severity)",
    "nvme": r"nvme.*(?:I/O error|timeout|reset)",
}

# Xid codes that mean hardware, not a customer's CUDA bug. Not exhaustive; these
# are the ones that change what you do next.
XID_HARDWARE = {
    48: "double-bit ECC error",
    63: "ECC page retirement or row remap recorded",
    64: "ECC page retirement or row remap failed",
    74: "NVLink error",
    79: "GPU fell off the bus",
    92: "high single-bit ECC correction rate",
    94: "contained ECC error",
    95: "uncontained ECC error",
}

SEVERITY_ORDER = {"info": 0, "warn": 1, "critical": 2}

# Seconds to wait on any external command before calling it hung. Long enough
# that a merely busy node is not reported as wedged, short enough that a sweep
# across a rack does not stall. --timeout overrides it.
DEFAULT_TIMEOUT = 20


class Finding:
    def __init__(self, layer, severity, summary, detail="", next_step="",
                 blocked=False):
        self.layer = layer
        self.severity = severity
        self.summary = summary
        self.detail = detail
        self.next_step = next_step
        # blocked means "this check did not run", which is not the same as
        # "this check found nothing". Conflating the two is how a triage tool
        # tells you a node is healthy when it never looked at it.
        self.blocked = blocked

    def as_dict(self):
        return {
            "layer": self.layer,
            "severity": self.severity,
            "summary": self.summary,
            "detail": self.detail,
            "next_step": self.next_step,
            "blocked": self.blocked,
        }


def run(cmd, timeout=None):
    """Run a command. Returns (rc, stdout, stderr). rc of -1 means timeout.

    nvidia-smi hangs when the driver is wedged, so everything gets a timeout and
    a timeout is treated as a finding rather than an error.
    """
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    if not shutil.which(cmd[0]):
        return (-2, "", f"{cmd[0]} not found")
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired:
        return (-1, "", f"{' '.join(cmd)} timed out after {timeout}s")


def check_kernel_log(findings):
    """Step 1. Passive, instant, and tells you which layer to look at next."""
    rc, out, err = run(["dmesg", "-T"])
    if rc == -2:
        findings.append(Finding("kernel", "info", "dmesg unavailable", err,
                                "install util-linux, or read journalctl -k",
                                blocked=True))
        return
    if rc != 0 and not out:
        findings.append(
            Finding("kernel", "info", "could not read kernel log", err,
                    "retry with sudo, or use journalctl -k", blocked=True)
        )
        return

    # Classify line by line rather than running each pattern over the whole
    # buffer. Two reasons: an Xid line also matches the bare NVRM pattern, so
    # whole-buffer counting reported every Xid twice, once properly and once as
    # an anonymous "NVRM message"; and the example in the report should be the
    # log line, not the fragment the regex happened to match. "example: NVRM:"
    # is not worth printing.
    xid_re = re.compile(KERNEL_PATTERNS["xid"])
    other = [(name, re.compile(p, re.IGNORECASE))
             for name, p in KERNEL_PATTERNS.items() if name != "xid"]

    xids = {}
    matched = {name: [] for name, _ in other}

    for line in out.splitlines():
        m = xid_re.search(line)
        if m:
            # Keep the first line for each code. A flapping link produces
            # hundreds of identical Xid 74s and the count is the useful part.
            xids.setdefault(int(m.group(1)), [line, 0])[1] += 1
            continue
        for name, pattern in other:
            if pattern.search(line):
                matched[name].append(line)
                break

    for xid in sorted(xids):
        example, count = xids[xid]
        seen = f"{count} occurrence(s). " if count > 1 else ""
        if xid in XID_HARDWARE:
            findings.append(Finding(
                "kernel", "critical",
                f"Xid {xid}: {XID_HARDWARE[xid]}",
                f"{seen}{example.strip()[:200]}",
                "Capture nvidia-bug-report.sh before any reset, then drain.",
            ))
        else:
            findings.append(Finding(
                "kernel", "warn",
                f"Xid {xid} present",
                f"{seen}Not in the hardware-class list, so it may be a "
                f"workload fault. {example.strip()[:160]}",
                "Correlate the timestamp against the failing job.",
            ))

    for name, _ in other:
        lines = matched[name]
        if lines:
            findings.append(Finding(
                "kernel", "warn",
                f"{len(lines)} {name} message(s) in kernel log",
                f"most recent: {lines[-1].strip()[:200]}",
                "Read the surrounding lines for timing against the symptom.",
            ))


def check_device_presence(findings, expected):
    """Step 2. lspci before nvidia-smi: a GPU off the bus hangs the driver tool.

    Returns True when nvidia-smi answered, which is what steps 3 and 4 need to
    know before they spend another timeout on the same wedged driver.
    """
    rc, out, err = run(["lspci"])
    bus_count = None
    if rc == 0:
        bus_count = len([ln for ln in out.splitlines()
                         if re.search(r"nvidia", ln, re.IGNORECASE)
                         and re.search(r"3d controller|vga", ln, re.IGNORECASE)])
        if bus_count != expected:
            findings.append(Finding(
                "presence", "critical",
                f"{bus_count} GPUs on the PCIe bus, expected {expected}",
                "A missing device on the bus is a hardware or seating fault, "
                "not a driver problem.",
                "Check BMC SEL for a PCIe event, then escalate for physical "
                "inspection.",
            ))
    else:
        findings.append(Finding("presence", "info", "lspci unavailable", err,
                                "install pciutils; the bus count is the only check that still answers when the driver is wedged", blocked=True))

    rc, out, err = run(["nvidia-smi", "-L"])
    if rc == -1:
        findings.append(Finding(
            "presence", "critical",
            "nvidia-smi timed out",
            err,
            "A wedged driver is itself the finding. Capture a bug report, "
            "then plan a reset window.",
        ))
        return False
    if rc == -2:
        findings.append(Finding("presence", "info", "nvidia-smi not found", err,
                                "no driver tooling on this host, so steps 3 and 4 cannot run", blocked=True))
        return False
    if rc != 0:
        findings.append(Finding(
            "presence", "critical",
            "nvidia-smi returned an error",
            err.strip()[:300],
            "Compare against lspci: driver-visible count differs from bus count.",
        ))
        return False

    driver_count = len([ln for ln in out.splitlines() if ln.startswith("GPU ")])
    if bus_count is not None and driver_count != bus_count:
        findings.append(Finding(
            "presence", "critical",
            f"driver sees {driver_count} GPUs, bus has {bus_count}",
            "The device is present but the driver is not binding it.",
            "Check dmesg for the bind failure before resetting.",
        ))
    elif driver_count != expected:
        findings.append(Finding(
            "presence", "critical",
            f"driver sees {driver_count} GPUs, expected {expected}",
            "", "Capture evidence, then drain.",
        ))
    return True


def _query(fields):
    """Query named fields. Returns a list of dicts keyed by field name.

    --query-gpu is stable across driver versions where the human-readable
    table is not. The catch is that it is all-or-nothing: one field the
    installed driver does not recognise and the whole query exits non-zero
    with "Field is not a valid field to query", taking the fields that would
    have worked with it. So the caller asks for one group at a time.
    """
    rc, out, _ = run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}",
         "--format=csv,noheader,nounits"]
    )
    if rc != 0:
        return None
    rows = []
    for line in out.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(fields):
            continue
        rows.append(dict(zip(fields, cells)))
    return rows


def _plural(count, noun):
    return noun if str(count) == "1" else noun + "s"


def _positive(value):
    """True for a counter that is present and above zero.

    Unsupported counters come back as "N/A" or "[N/A]" rather than an error,
    and "0" is the common case, so neither should be mistaken for a fault.
    """
    return value.isdigit() and int(value) > 0


# Split by generation support, not by subject matter. ECC and thermals answer
# everywhere; remapped_rows is Ampere and later, and asking for it on a V100
# would fail the ECC fields sitting next to it.
CORE_FIELDS = [
    "index",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.total",
    "temperature.gpu",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
]
REMAP_FIELDS = [
    "index",
    "remapped_rows.pending",
    "remapped_rows.failure",
    "remapped_rows.uncorrectable",
]


def check_gpu_state(findings):
    """Step 3. ECC, row remapping, throttling, thermals."""
    rows = _query(CORE_FIELDS)
    if rows is None:
        findings.append(Finding(
            "gpu_state", "info", "could not query ECC and thermal state", "",
            "nvidia-smi -q -d ECC,PERFORMANCE,TEMPERATURE by hand.",
            blocked=True,
        ))
    for r in rows or []:
        idx = r["index"]
        vol = r["ecc.errors.uncorrected.volatile.total"]
        agg = r["ecc.errors.uncorrected.aggregate.total"]
        if _positive(vol):
            findings.append(Finding(
                "gpu_state", "critical",
                f"GPU {idx}: {vol} uncorrectable ECC "
                f"{_plural(vol, 'error')} (volatile)",
                "Volatile counters reset on driver reload, so this is recent.",
                "Capture a bug report before any reset clears it.",
            ))
        if _positive(agg) and not _positive(vol):
            findings.append(Finding(
                "gpu_state", "warn",
                f"GPU {idx}: {agg} uncorrectable ECC "
                f"{_plural(agg, 'error')} (aggregate only)",
                "A history of uncorrectable errors, nothing since the last "
                "driver reload.",
                "Note it against the node and compare with a healthy neighbour.",
            ))
        for field, label in (
            ("clocks_throttle_reasons.hw_slowdown", "hardware slowdown"),
            ("clocks_throttle_reasons.sw_thermal_slowdown",
             "software thermal slowdown"),
        ):
            if r[field] == "Active":
                findings.append(Finding(
                    "gpu_state", "warn",
                    f"GPU {idx}: {label} active",
                    f"temperature {r['temperature.gpu']}C",
                    "Thermal or power event. Check the BMC SEL and inlet temps.",
                ))

    _check_row_remapping(findings)


def _check_row_remapping(findings):
    """Row remapping is Ampere and later, and it is not the same as ECC.

    A pending remap means the GPU wants a reset to apply one. A failure means
    it tried and could not, which is an RMA conversation rather than a reset.
    """
    rows = _query(REMAP_FIELDS)
    if rows is not None:
        for r in rows:
            idx = r["index"]
            if r["remapped_rows.failure"] in ("Yes", "1"):
                findings.append(Finding(
                    "gpu_state", "critical",
                    f"GPU {idx}: row remapping failure",
                    "The GPU attempted a remap and could not complete it.",
                    "Capture evidence and open an RMA.",
                ))
            if r["remapped_rows.pending"] in ("Yes", "1"):
                findings.append(Finding(
                    "gpu_state", "warn",
                    f"GPU {idx}: row remap pending",
                    "A reset is needed to apply it.",
                    "Schedule a reset window; do not reset mid-job.",
                ))
            if _positive(r["remapped_rows.uncorrectable"]):
                findings.append(Finding(
                    "gpu_state", "warn",
                    f"GPU {idx}: {r['remapped_rows.uncorrectable']} rows "
                    "remapped from uncorrectable errors",
                    "Remapping is finite. A climbing count is a dying board.",
                    "Trend it against the rest of the fleet.",
                ))
        return

    # Pre-Ampere, or a driver that does not expose the fields. Fall back to the
    # -q section, which does not report per-GPU in a form worth parsing, so
    # this only answers whether the node has a problem at all.
    rc, out, _ = run(["nvidia-smi", "-q", "-d", "ROW_REMAPPER"])
    if rc != 0:
        findings.append(Finding(
            "gpu_state", "info", "could not read row remapping state", "",
            "Expected on pre-Ampere hardware, where page retirement applies "
            "instead: nvidia-smi -q -d PAGE_RETIREMENT.",
            blocked=True,
        ))
        return
    if re.search(r"Remapping Failure Occurred\s*:\s*Yes", out):
        findings.append(Finding(
            "gpu_state", "critical",
            "row remapping failure on at least one GPU",
            "The GPU attempted a remap and could not complete it.",
            "nvidia-smi -q -d ROW_REMAPPER to find which, then open an RMA.",
        ))
    if re.search(r"Pending\s*:\s*Yes", out):
        findings.append(Finding(
            "gpu_state", "warn",
            "row remap pending on at least one GPU",
            "", "Reset required to apply. Schedule a window.",
        ))


def check_nvlink(findings):
    """Step 4. Links down or error counters climbing."""
    rc, out, err = run(["nvidia-smi", "nvlink", "-s"])
    if rc == -2:
        findings.append(Finding("nvlink", "info", "nvlink status unavailable",
                                err, "", blocked=True))
        return
    if rc != 0:
        findings.append(Finding("nvlink", "info", "nvlink status unavailable", err,
                                "", blocked=True))
        return
    inactive = [ln.strip() for ln in out.splitlines() if "inactive" in ln.lower()]
    if inactive:
        findings.append(Finding(
            "nvlink", "critical",
            f"{len(inactive)} NVLink(s) inactive",
            f"example: {inactive[0][:160]}",
            "A down link degrades or hangs collectives. Compare topology "
            "against a healthy node.",
        ))


def check_fabric(findings):
    """Step 5. Move this first when the symptom is a hang or a slow multi-node job."""
    rc, out, err = run(["ibstat"])
    if rc == -2:
        findings.append(Finding("fabric", "info", "ibstat not found", err,
                                "install infiniband-diags, or pass --no-fabric if this host has no fabric", blocked=True))
        return
    if rc != 0:
        findings.append(Finding("fabric", "info", "ibstat returned an error", err,
                                "", blocked=True))
        return

    down = re.findall(r"State:\s*(\w+)", out)
    bad = [s for s in down if s.lower() != "active"]
    if bad:
        findings.append(Finding(
            "fabric", "critical",
            f"{len(bad)} InfiniBand port(s) not Active: {', '.join(set(bad))}",
            "", "Check the switch side and the transceiver before blaming the host.",
        ))

    rates = re.findall(r"Rate:\s*(\d+)", out)
    if rates:
        distinct = sorted({int(r) for r in rates})
        if len(distinct) > 1:
            findings.append(Finding(
                "fabric", "warn",
                f"mixed link rates across ports: {distinct}",
                "A port negotiated below the others.",
                "mlxlink -d <dev> for the negotiated vs supported speed.",
            ))


def check_oob(findings):
    """Step 6. Events the OS never saw. Start here if the host is unresponsive."""
    rc, out, err = run(["ipmitool", "sel", "list", "last", "20"])
    if rc == -2:
        findings.append(Finding("oob", "info", "ipmitool not found", err,
                                "install ipmitool, or pass --no-oob", blocked=True))
        return
    if rc != 0:
        findings.append(Finding("oob", "info", "could not read BMC SEL", err,
                                "needs root or BMC credentials", blocked=True))
        return
    interesting = [
        ln for ln in out.splitlines()
        if re.search(r"critical|non-recoverable|correctable|thermal|power supply",
                     ln, re.IGNORECASE)
    ]
    if interesting:
        findings.append(Finding(
            "oob", "warn",
            f"{len(interesting)} notable BMC SEL event(s)",
            f"most recent: {interesting[-1][:160]}",
            "Correlate the SEL timestamp against the symptom.",
        ))


LAYER_LABELS = [
    ("kernel", "1. Kernel log"),
    ("presence", "2. Device presence"),
    ("gpu_state", "3. GPU state"),
    ("nvlink", "4. NVLink"),
    ("fabric", "5. Fabric"),
    ("oob", "6. Out of band"),
]


def exit_code(findings):
    """1 if something needs a human, 2 if the sweep was incomplete, else 0.

    A real finding outranks an incomplete sweep: if a check found an
    uncorrectable ECC error, the missing ipmitool is not the headline.
    """
    if any(f.severity in ("warn", "critical") for f in findings):
        return 1
    if any(f.blocked for f in findings):
        return 2
    return 0


def report(findings, skipped=frozenset()):
    by_layer = {k: [] for k, _ in LAYER_LABELS}
    for f in findings:
        by_layer.setdefault(f.layer, []).append(f)

    for key, label in LAYER_LABELS:
        items = by_layer.get(key, [])
        print(f"\n{label}")
        if key in skipped:
            print("   not checked (disabled by flag)")
            continue
        if not items:
            print("   nothing found")
            continue
        for f in items:
            marker = {"info": "   ", "warn": " ! ", "critical": " X "}[f.severity]
            if f.blocked:
                marker = " ? "
            print(f"{marker}{f.summary}")
            if f.detail:
                print(f"     {f.detail}")
            if f.next_step:
                print(f"     next: {f.next_step}")

    print("\n" + "-" * 68)
    actionable = [f for f in findings if f.severity in ("warn", "critical")]
    blocked = [f for f in findings if f.blocked]

    if actionable:
        print(f"{len(actionable)} finding(s) need a human.")
        print("Capture nvidia-bug-report.sh before anything that resets state,")
        print("then run the same sweep on a healthy node in the same rack.")
        if blocked:
            print(f"\n{len(blocked)} check(s) did not run, so this is not the "
                  "whole picture.")
    elif blocked:
        names = ", ".join(sorted({f.layer for f in blocked}))
        print(f"Sweep incomplete: {len(blocked)} check(s) could not run "
              f"({names}).")
        print("Nothing was found, but nothing was ruled out either. Fix the")
        print("tooling or rerun with sudo before you clear this node.")
    else:
        print("No findings. If the symptom is real, compare against a healthy")
        print("neighbour (step 9) before clearing the node.")


def main():
    global DEFAULT_TIMEOUT
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect-gpus", type=int, default=8,
                    help="expected GPU count (default 8; use 4 for a GB300 tray)")
    ap.add_argument("--no-fabric", action="store_true",
                    help="skip InfiniBand checks")
    ap.add_argument("--no-oob", action="store_true",
                    help="skip BMC SEL checks")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"seconds before a command is called hung "
                         f"(default {DEFAULT_TIMEOUT})")
    args = ap.parse_args()
    DEFAULT_TIMEOUT = args.timeout

    findings = []
    # A check turned off by a flag is not a check that found nothing, and it is
    # not a check that could not run either. Third case, said out loud.
    skipped = set()
    check_kernel_log(findings)
    driver_responding = check_device_presence(findings, args.expect_gpus)
    if driver_responding:
        check_gpu_state(findings)
        check_nvlink(findings)
    else:
        # Step 2 already recorded why. Repeating the same call in steps 3 and 4
        # buys nothing and costs another two timeouts on a node someone is
        # waiting on.
        for layer in ("gpu_state", "nvlink"):
            findings.append(Finding(
                layer, "info", "skipped: nvidia-smi is not usable on this host",
                "", "See step 2.", blocked=True,
            ))
    if args.no_fabric:
        skipped.add("fabric")
    else:
        check_fabric(findings)
    if args.no_oob:
        skipped.add("oob")
    else:
        check_oob(findings)

    if args.json:
        print(json.dumps({
            "complete": not any(f.blocked for f in findings),
            "skipped_by_request": sorted(skipped),
            "findings": [f.as_dict() for f in findings],
        }, indent=2))
    else:
        report(findings, skipped)

    return exit_code(findings)


if __name__ == "__main__":
    sys.exit(main())
