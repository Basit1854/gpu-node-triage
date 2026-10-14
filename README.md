# gpu-node-triage

A read-only triage tool for suspect GPU nodes in AI training clusters, and the
ordering logic behind it.

The script is the smaller half of this repo. The useful part is the **order**,
because getting that wrong is how you destroy the evidence you needed.

---

## Why ordering matters more than tooling

Two mistakes account for most wasted time on a bad GPU node.

**Running active tests before reading passive ones.** `dcgmi diag` at level 2 or
3 is a stress test. It takes minutes, needs the GPUs idle, and changes the
node's state while it runs. `dmesg` is instant, free, and tells you what already
happened. Read the history before you run the tests.

**Resetting before capturing.** A reboot or GPU reset wipes the kernel ring
buffer and the volatile ECC counters. Those are frequently the only proof that
the fault was real. Capture first, reset second.

There is a third, subtler one: reaching for `nvidia-smi` to count GPUs. If a GPU
has fallen off the PCIe bus, `nvidia-smi` can hang or error out while `lspci`
still answers. Count on the bus, then ask the driver.

---

## The order

| # | Step | Command | Why here |
|---|------|---------|----------|
| 0 | **Contain** | `kubectl cordon` + `drain`, or `scontrol update nodename=<n> state=DRAIN` | Stop new jobs landing on a node you suspect. Note the symptom and when it started. |
| 1 | **Kernel log** | `dmesg -T \| grep -iE 'xid\|nvrm\|mlx5\|aer\|nvme'` | Passive and instant. An Xid code points you at the right layer before you touch anything. |
| 2 | **Device presence** | `lspci \| grep -i nvidia \| wc -l`, then `nvidia-smi -L` | Expect 8 on HGX, 4 per GB300 compute tray. `nvidia-smi` hanging is itself a finding. |
| 3 | **GPU state** | `nvidia-smi -q -d ECC,ROW_REMAPPER,PERFORMANCE,TEMPERATURE,POWER` | Uncorrectable ECC, remap pending/failed, throttle reasons, thermals, power. |
| 4 | **NVLink** | `nvidia-smi nvlink -s`, `nvidia-smi nvlink -e` | Links down or error counters climbing. |
| 5 | **Fabric** | `ibstat`, `ibdev2netdev`, `mlxlink -d <dev>` | Link state, negotiated speed, error counters. **Move this to step 1 if the symptom is a hang or a slow multi-node job** — a collective stalls on one bad link. |
| 6 | **Out of band** | `ipmitool sel list`, Redfish sensors | Power, thermal and PCIe events the OS never saw. **Start here if the host is unresponsive.** |
| 7 | **Capture evidence** | `nvidia-bug-report.sh` | Before anything that resets state. |
| 8 | **Active tests** | `dcgmi diag -r 1` (seconds), then `-r 2` / `-r 3` (minutes, GPUs idle) | Finds faults the logs never caught. Last because it is invasive. |
| 9 | **Compare and decide** | Same checks on a healthy neighbour | Turns "looks odd" into proof. Then reset, power cycle, or RMA. |

Step 9 is the one people skip. A counter you have never looked at before always
looks alarming. Running the same checks against a known-good node in the same
rack is what separates a real fault from a number you have not seen.

---

## What the script does

`gpu-triage.py` runs **steps 1 through 6, read-only**, with no dependencies
outside the Python standard library, because installing packages on a suspect
node is not triage.

It will not cordon, will not reset, will not run `dcgmi diag`, and will not
reboot anything. Those are decisions for a human with context, and two of them
destroy evidence.

```bash
# default: read-only sweep, human-readable output
sudo ./gpu-triage.py

# expect 4 GPUs instead of 8 (GB300 compute tray)
sudo ./gpu-triage.py --expect-gpus 4

# machine-readable, for piping into a ticket or a runbook step
sudo ./gpu-triage.py --json

# skip the fabric checks on a single-node box
sudo ./gpu-triage.py --no-fabric

# a loaded node answers slowly; a rack sweep cannot wait 20s per command
sudo ./gpu-triage.py --timeout 45
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Every check ran and found nothing. |
| `1` | Something needs a human. |
| `2` | The sweep could not complete: missing tooling or insufficient privileges. |

`2` matters more than it looks. A check that could not run is not a check that
found nothing, and anything wrapping this in a health check needs to tell those
apart. Missing `nvidia-smi` used to print "nvidia-smi not found" and then
conclude "No findings", exit `0`, which reads as a clean node to everything
downstream.

A real finding outranks an incomplete sweep: if there is an uncorrectable ECC
error, a missing `ipmitool` is not the headline, so that case exits `1` and says
separately that some checks were skipped.

Output is grouped by layer, in the order above, with a short verdict at the end
naming the next step rather than the conclusion. "Xid 48 present, GPU 3,
double-bit ECC, capture a bug report then drain" is useful. "Node unhealthy" is
not.

### Design decisions worth naming

**Every external command has a timeout.** A wedged driver hangs `nvidia-smi`,
and a triage tool that hangs is worse than no tool. A timeout is reported as a
finding, not an error: a driver that will not answer is itself the answer.

**The driver is probed once.** Step 2 establishes whether `nvidia-smi` responds,
and steps 3 and 4 run only if it did. They used to call it again regardless, so
one hung node cost three timeouts while somebody waited on the output.

**Nothing is parsed from `nvidia-smi` table output.** The human-readable table
changes between driver versions. `--query-gpu` with explicit fields is stable.

**Query fields are grouped by generation support.** `--query-gpu` is
all-or-nothing: one field the installed driver does not recognise and the whole
query exits non-zero, taking the fields that would have worked alongside it. ECC
and thermals answer everywhere; `remapped_rows` is Ampere and later. Asked for
together, a V100 loses all of step 3.

**Volatile and aggregate ECC counters are reported separately.** Volatile resets
on driver reload, aggregate does not. Errors in aggregate alone are history
rather than something happening now, which is a different conversation, so that
case is a warning and not a critical.

**Row remapping is checked, and it is not page retirement.** Row remapping
replaced page retirement on A100 and later, so `retired_pages.pending` returns
`N/A` on anything current. Pending means the GPU wants a reset to apply a remap,
which is a maintenance window. Failure means it tried and could not, which is an
RMA. Reporting them together sends working boards back to the vendor.

**Severity is advisory.** The script flags and explains; it does not decide.
Deciding needs the comparison in step 9 and knowledge of what the customer is
running.

---

## Tests

```bash
python3 tests/test_triage.py
```

There is no GPU on a laptop or a CI runner, so the tests put stub `nvidia-smi`
and `lspci` binaries ahead of the real ones on `PATH` and shape their output the
way the real tools shape it. The stubs in `tests/fake-bin` are the interesting
half: they reproduce the failure modes that are easy to get wrong by reasoning
about them rather than observing them, including an unsupported query field
exiting non-zero instead of returning `N/A`, and a driver that never returns at
all.

---

## Not covered

Burn-in and acceptance testing for new nodes, performance baselining, and
NCCL-level collective debugging. Different problem: those are about proving a
node is good, this is about finding out why one went bad.
