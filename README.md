# ChampSim

![GitHub](https://img.shields.io/github/license/ChampSim/ChampSim)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ChampSim/ChampSim/test.yml)
![GitHub forks](https://img.shields.io/github/forks/ChampSim/ChampSim)
[![Coverage Status](https://coveralls.io/repos/github/ChampSim/ChampSim/badge.svg?branch=develop)](https://coveralls.io/github/ChampSim/ChampSim?branch=develop)

ChampSim is a trace-based simulator for a microarchitecture study. If you have questions about how to use ChampSim, we encourage you to search the threads in the Discussions tab or start your own thread. If you are aware of a bug or have a feature request, open a new Issue.

# Using ChampSim

ChampSim is the result of academic research. If you use this software in your work, please cite it using the following reference:

    Gober, N., Chacon, G., Wang, L., Gratz, P. V., Jimenez, D. A., Teran, E., Pugsley, S., & Kim, J. (2022). The Championship Simulator: Architectural Simulation for Education and Competition. https://doi.org/10.48550/arXiv.2210.14324

If you use ChampSim in your work, you may submit a pull request modifying `PUBLICATIONS_USING_CHAMPSIM.bib` to have it featured in [the documentation](https://champsim.github.io/ChampSim/master/Publications-using-champsim.html).

# Download dependencies

ChampSim uses [vcpkg](https://vcpkg.io) to manage its dependencies. In this repository, vcpkg is included as a submodule. You can download the dependencies with
```
git submodule update --init
vcpkg/bootstrap-vcpkg.sh
vcpkg/vcpkg install
```

# Compile

Pass a JSON configuration to the default Make target. Make runs `config.sh` first and builds the executables named by that configuration.

```sh
make -j8 CONFIG=champsim_config/champsim_config.json
```

On a fresh checkout, plain `make` uses `champsim_config/champsim_config.json`. If you have already generated a configuration, plain `make` uses it. You can also configure manually, including running `./config.sh` with no arguments to use the simulator defaults.
```
$ ./config.sh <configuration file>
$ make
```

# Download DPC-3 trace

Traces used for the 3rd Data Prefetching Championship (DPC-3) can be found here. (https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/) A set of traces used for the 2nd Cache Replacement Championship (CRC-2) can be found from this link. (http://bit.ly/2t2nkUj)

Storage for these traces is kindly provided by Daniel Jimenez (Texas A&M University) and Mike Ferdman (Stony Brook University). If you find yourself frequently using ChampSim, it is highly encouraged that you maintain your own repository of traces, in case the links ever break.

# Run simulation

Build with `CONFIG`, then select an executable under `bin/` using `PROGRAM`. Use `TRACE` with `run` for one trace and `TRACE_DIR` with `run-all` for a batch.

```sh
make -j8 CONFIG=champsim_config/Triage-sms.json
make run PROGRAM=triage-sms-4way TRACE=471.omnetpp-188B.champsimtrace.xz
make run-all PROGRAM=triage-sms-4way TRACE_DIR="$HOME/data/SPEC2006" JOBS=16
```

Both targets write logs to `log/<PROGRAM>/` by default. The single-trace target displays the output and writes a uniquely named `<trace-file>.<id>.log`. The batch target skips a trace if its exact `<trace-file>.log` exists. Failed runs leave `.partial` logs, which can be retried. Set a fresh `LOG_DIR` for an independent batch, including when changing the instruction counts.

You can also pass `CONFIG` directly to `run` or `run-all` to build and run in one command. Set `PROGRAM` to select an executable from that configuration. For a single-configuration input with no `PROGRAM`, this workflow names the executable after the final JSON file.

You can also invoke the executable directly.
```
$ bin/champsim --warmup-instructions 200000000 --simulation-instructions 500000000 ~/path/to/traces/600.perlbench_s-210B.champsimtrace.xz
```

The number of warmup and simulation instructions given will be the number of instructions retired. Note that the statistics printed at the end of the simulation include only the simulation phase.

## Triage and SMS experiments

Build the original four L2 prefetcher variants and the current-PC Lookahead variant with the same base configuration and four LLC metadata ways.

```sh
make -j8 CONFIG=champsim_config/Triage-sms-modes.json
make -j8 CONFIG=champsim_config/Triage-sms-lookahead-current-pc.json
```

The Triage JSON files are self-contained. Use `Triage-sms.json` to build only the SMS-filter variant, `Triage-dynamic.json` for dynamic LLC metadata partitioning, and `Triage-ways.json` for the fixed metadata-way sweep.

| Binary | L2 prefetcher class | Behavior |
| --- | --- | --- |
| `triage-4way` | `triage` | Triage with its original training stream |
| `triage-sms-4way` | `triage_sms` | SMS filters Triage training |
| `triage-sms-spatial-4way` | `triage_sms_spatial` | Demand-triggered SMS shares the L2 prefetch queue with Triage |
| `triage-sms-lookahead-4way` | `triage_sms_lookahead` | Uses an exact cross-region transition table to recover the historical target PC for stable footprint replay |
| `triage-sms-lookahead-current-pc-4way` | `triage_sms_lookahead_current_pc` | Uses the current demand PC and predicted target offset for stable footprint replay across regions |

Edit `L2C.prefetcher` to select a class. The spatial and Lookahead variants accept `L2C.triage_sms_degree` from 1 to 63, with a default of 4 candidates per pattern expansion. Re-run `make CONFIG=...` after changing the JSON.

The spatial and Lookahead variants use a 512-entry pattern history table. Both Lookahead variants replay the intersection of the two most recent completed footprints for the selected PC and offset. The original Lookahead allocates an 8,192-entry exact transition table. The current-PC variant carries the demand PC with each pending Triage prediction. Both report speculative requests under `sms_lookahead` in the L2 JSON statistics, with the variant identified by `triage prefetch.mode`.

# Add your own branch predictor, data prefetchers, and replacement policy
**Copy an empty template**
```
$ mkdir prefetcher/mypref
$ cp prefetcher/no_l2c/no.cc prefetcher/mypref/mypref.cc
```

**Work on your algorithms with your favorite text editor**
```
$ vim prefetcher/mypref/mypref.cc
```

**Compile and test**
Add your prefetcher to the configuration file.
```
{
    "L2C": {
        "prefetcher": "mypref"
    }
}
```
Note that the example prefetcher is an L2 prefetcher. You might design a prefetcher for a different level.

```
$ ./config.sh <configuration file>
$ make
$ bin/champsim --warmup-instructions 200000000 --simulation-instructions 500000000 600.perlbench_s-210B.champsimtrace.xz
```

# How to create traces

Program traces are available in a variety of locations, however, many ChampSim users wish to trace their own programs for research purposes.
Example tracing utilities are provided in the `tracer/` directory.

# Evaluate Simulation

ChampSim measures the IPC (Instruction Per Cycle) value as a performance metric. <br>
There are some other useful metrics printed out at the end of simulation. <br>

Good luck and be a champion! <br>
