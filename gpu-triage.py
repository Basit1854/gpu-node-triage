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
    "nvrm": r"NVRM:",
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


class Finding:
    def __init__(self, layer, severity, summary, detail="", next_step=""):
        self.layer = layer
        self.severity = severity
        self.summary = summary
        self.detail = detail
        self.next_step = next_step

    def as_dict(self):
        return {
            "layer": self.layer,
            "severity": self.severity,
            "summary": self.summary,
            "detail": self.detail,
            "next_step": self.next_step,
        }


def run(cmd, timeout=15):
    """Run a command. Returns (rc, stdout, stderr). rc of -1 means timeout.

    nvidia-smi hangs when the driver is wedged, so everything gets a timeout and
    a timeout is treated as a finding rather than an error.
    """
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
    rc, out, err = run(["dmesg", "-T"], timeout=20)
    if rc == -2:
        findings.append(Finding("kernel", "info", "dmesg unavailable", err))
        return
    if rc != 0 and not out:
        findings.append(
            Finding("kernel", "info", "could not read kernel log", err,
                    "retry with sudo, or use journalctl -k")
        )
        return

    xids = sorted({int(m) for m in re.findall(KERNEL_PATTERNS["xid"], out)})
    for xid in xids:
        if xid in XID_HARDWARE:
            findings.append(Finding(
                "kernel", "critical",
                f"Xid {xid}: {XID_HARDWARE[xid]}",
                "Hardware-class Xid present in the kernel log.",
                "Capture nvidia-bug-report.sh before any reset, then drain.",
            ))
        else:
            findings.append(Finding(
                "kernel", "warn",
                f"Xid {xid} present",
                "Not in the hardware-class list; may be a workload fault.",
                "Correlate the timestamp against the failing job.",
            ))

    for name, pattern in KERNEL_PATTERNS.items():
        if name == "xid":
            continue
        hits = re.findall(pattern, out, flags=re.IGNORECASE)
        if hits:
            findings.append(Finding(
                "kernel", "warn",
                f"{len(hits)} {name} message(s) in kernel log",
                f"example: {hits[0][:160]}",
                "Read the surrounding lines for timing against the symptom.",
            ))


def check_device_presence(findings, expected):
    """Step 2. lspci before nvidia-smi: a GPU off the bus hangs the driver tool."""
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
        findings.append(Finding("presence", "info", "lspci unavailable", err))

    rc, out, err = run(["nvidia-smi", "-L"], timeout=20)
    if rc == -1:
        findings.append(Finding(
            "presence", "critical",
            "nvidia-smi timed out",
            err,
            "A wedged driver is itself the finding. Capture a bug report, "
            "then plan a reset window.",
        ))
        return bus_count, None
    if rc == -2:
        findings.append(Finding("presence", "info", "nvidia-smi not found", err))
        return bus_count, None
    if rc != 0:
        findings.append(Finding(
            "presence", "critical",
            "nvidia-smi returned an error",
            err.strip()[:300],
            "Compare against lspci: driver-visible count differs from bus count.",
        ))
        return bus_count, None

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
    return bus_count, driver_count


def _query(fields, timeout=20):
    """nvidia-smi --query-gpu is stable across driver versions. The table is not."""
    rc, out, err = run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}",
         "--format=csv,noheader,nounits"], timeout=timeout
    )
    if rc != 0:
        return None
    rows = []
    for line in out.strip().splitlines():
        rows.append([c.strip() for c in line.split(",")])
    return rows


def check_gpu_state(findings):
    """Step 3. ECC, row remapping, throttling, thermals, power."""
    rows = _query([
        "index",
        "ecc.errors.uncorrected.volatile.total",
        "ecc.errors.uncorrected.aggregate.total",
        "retired_pages.pending",
        "temperature.gpu",
        "clocks_throttle_reasons.hw_slowdown",
        "clocks_throttle_reasons.sw_thermal_slowdown",
    ])
    if rows is None:
        findings.append(Finding(
            "gpu_state", "info", "could not query GPU state", "",
            "nvidia-smi -q -d ECC,ROW_REMAPPER,PERFORMANCE by hand.",
        ))
    else:
        for r in rows:
            idx = r[0]
            vol, agg, pending = r[1], r[2], r[3]
            if vol.isdigit() and int(vol) > 0:
                findings.append(Finding(
                    "gpu_state", "critical",
                    f"GPU {idx}: {vol} uncorrectable ECC errors (volatile)",
                    "Volatile counters reset on driver reload, so this is recent.",
                    "Capture a bug report before any reset clears it.",
                ))
            if agg.isdigit() and int(agg) > 0 and vol in ("0", "N/A"):
                findings.append(Finding(
                    "gpu_state", "warn",
                    f"GPU {idx}: {agg} uncorrectable ECC errors (aggregate only)",
                    "History of uncorrectable errors, nothing since last reload.",
                    "Note it against the node; compare with a healthy neighbour.",
                ))
            if pending not in ("0", "N/A", ""):
                findings.append(Finding(
                    "gpu_state", "warn",
                    f"GPU {idx}: {pending} retired pages pending",
                    "A reset is needed to apply the retirement.",
                    "Schedule a reset window; do not reset mid-job.",
                ))
            if len(r) > 5 and r[5] == "Active":
                findings.append(Finding(
                    "gpu_state", "warn",
                    f"GPU {idx}: hardware slowdown active",
                    f"temperature {r[4]}C",
                    "Thermal or power event. Check BMC SEL and inlet temps.",
                ))

    # Row remapping lives in -q output, not the query interface.
    rc, out, _ = run(["nvidia-smi", "-q", "-d", "ROW_REMAPPER"], timeout=20)
    if rc == 0:
        if re.search(r"Remapping Failure Occurred\s*:\s*Yes", out):
            findings.append(Finding(
                "gpu_state", "critical",
                "row remapping failure occurred",
                "The GPU attempted a remap and could not complete it.",
                "Capture evidence and open an RMA.",
            ))
        if re.search(r"Pending\s*:\s*Yes", out):
            findings.append(Finding(
                "gpu_state", "warn",
                "row remapping pending",
                "", "Reset required to apply. Schedule a window.",
            ))


def check_nvlink(findings):
    """Step 4. Links down or error counters climbing."""
    rc, out, err = run(["nvidia-smi", "nvlink", "-s"], timeout=20)
    if rc == -2:
        return
    if rc != 0:
        findings.append(Finding("nvlink", "info", "nvlink status unavailable", err))
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
    rc, out, err = run(["ibstat"], timeout=20)
    if rc == -2:
        findings.append(Finding("fabric", "info", "ibstat not found", err))
        return
    if rc != 0:
        findings.append(Finding("fabric", "info", "ibstat returned an error", err))
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
    rc, out, err = run(["ipmitool", "sel", "list", "last", "20"], timeout=25)
    if rc == -2:
        findings.append(Finding("oob", "info", "ipmitool not found", err))
        return
    if rc != 0:
        findings.append(Finding("oob", "info", "could not read BMC SEL", err))
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


def report(findings):
    worst = "info"
    by_layer = {k: [] for k, _ in LAYER_LABELS}
    for f in findings:
        by_layer.setdefault(f.layer, []).append(f)
        if SEVERITY_ORDER[f.severity] > SEVERITY_ORDER[worst]:
            worst = f.severity

    for key, label in LAYER_LABELS:
        items = by_layer.get(key, [])
        print(f"\n{label}")
        if not items:
            print("   nothing found")
            continue
        for f in items:
            marker = {"info": "   ", "warn": " ! ", "critical": " X "}[f.severity]
            print(f"{marker}{f.summary}")
            if f.detail:
                print(f"     {f.detail}")
            if f.next_step:
                print(f"     next: {f.next_step}")

    print("\n" + "-" * 68)
    actionable = [f for f in findings if f.severity in ("warn", "critical")]
    if not actionable:
        print("No findings. If the symptom is real, compare against a healthy")
        print("neighbour (step 9) before clearing the node.")
    else:
        print(f"{len(actionable)} finding(s) need a human.")
        print("Capture nvidia-bug-report.sh before anything that resets state,")
        print("then run the same sweep on a healthy node in the same rack.")
    return worst


def main():
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
    args = ap.parse_args()

    findings = []
    check_kernel_log(findings)
    check_device_presence(findings, args.expect_gpus)
    check_gpu_state(findings)
    check_nvlink(findings)
    if not args.no_fabric:
        check_fabric(findings)
    if not args.no_oob:
        check_oob(findings)

    if args.json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
        worst = max((f.severity for f in findings),
                    key=lambda s: SEVERITY_ORDER[s], default="info")
    else:
        worst = report(findings)

    return 1 if worst in ("warn", "critical") else 0


if __name__ == "__main__":
    sys.exit(main())
