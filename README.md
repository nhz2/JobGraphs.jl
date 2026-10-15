# JobGraphs.jl

[![CI](https://github.com/nhz2/JobGraphs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/nhz2/JobGraphs.jl/actions/workflows/CI.yml)

A Julia package to help plan multi-threaded work.

You describe your work up front as a graph of chunked jobs and their dependencies.
JobGraphs then runs it on a fixed set of worker threads.
A job starts as soon as the jobs it depends on finish, and independent jobs overlap.

It works best for programs whose schedule is mostly the same on every run, such as the
inner loop of a simulation or solver.

## Installation

The package is not registered yet. It requires Julia 1.12 or later.

```julia
using Pkg
Pkg.add(url="https://github.com/nhz2/JobGraphs.jl")
```

Start Julia with several threads, for example `julia --threads=auto`. Otherwise
there is only one worker.

## Quick start

This example also uses [ChunkSplitters.jl](https://github.com/JuliaFolds2/ChunkSplitters.jl)
to split the input into chunks.

```julia
using JobGraphs
using ChunkSplitters: chunks

x = randn(1_000_000)
xchunks = chunks(x; n=16)  # xchunks[i] is a view of the i-th chunk of x
nworkers = Threads.nthreads()
partials = zeros(nworkers)  # one running max per worker
result = Ref(0.0)

g = JobGraph([
    :reset => (; job=Job(1) do i, info
        fill!(partials, -Inf)
        nothing
    end),
    :partial => (; job=Job(length(xchunks)) do i, info
        w = info.worker_idx
        partials[w] = max(partials[w], maximum(abs2, xchunks[i]))
        nothing
    end, deps=:reset),
    :max => (; job=Job(1) do i, info
        result[] = maximum(partials)
        nothing
    end, deps=:partial),
])

p = plan(g, nworkers)  # build once
run!(p)                # run as many times as you like
result[] == maximum(abs2, x)  # true
```

Each chunk updates the running max of the worker running it,
`partials[info.worker_idx]`. A worker runs one chunk at a time, so this needs no
locks or atomics. `:reset` clears the running maxes at the start of each run.

There are three steps:

1. **Declare** a `JobGraph`: a list of labeled nodes, each a `Job` (or a nested
   `JobGraph`) with optional dependencies and predicates.
2. **Plan** it with `plan(g, nworkers)`.
3. **Run** it with `run!(p)`. You can run a plan many times, but not concurrently.

## Concepts

### `Job(fun, nchunks)`

A job is a parallel loop. `fun(chunk_idx, info)` is called once for each
`chunk_idx in 1:nchunks`, on any worker. `fun` captures its inputs and outputs, and
its return value is ignored. You decide how big a chunk is. Use enough chunks to
balance the load, but make each one large enough that its work outweighs the overhead.

`info` is a `WorkerInfo`. Its field `info.worker_idx` is in `1:nworkers` and
identifies the calling worker. Use it to index per-worker scratch space:

```julia
scratch = [zeros(100) for _ in 1:nworkers]
Job(nchunks) do i, info
    buf = scratch[info.worker_idx]
    # ... use buf ...
end
```

A job with zero chunks is pruned from the graph. Dependencies through it are kept.

### `JobGraph(nodes; repeat_if=nothing)`

A graph is a vector of `label => (; job, deps, do_if | skip_chunks)` pairs:

- `job`: a `Job` or a nested `JobGraph`.
- `deps`: the labels of **earlier** nodes this node depends on, either as a
  collection or as a single `Symbol`. Every chunk of every dependency finishes
  before any chunk of this node starts, so a chunk can read what its dependencies
  wrote without extra synchronization.
- `do_if(info)::Bool` (optional): evaluated each time the node becomes ready. If it
  returns `false`, the node is skipped, and its dependents proceed as if it had run.
- `skip_chunks(info)::Int` (optional): a finer version of `do_if` that returns how
  many leading chunks to skip. `0` runs every chunk, and anything `≥ nchunks`, such
  as `typemax(Int)`, skips the whole node. A node can have `do_if` or `skip_chunks`,
  but not both.

Declaration order is a soft priority, so put long or critical-path jobs first.

`repeat_if(info)::Bool` is called after every node of the graph has finished. If it
returns `true`, the graph runs again from the start and its `do_if` and
`skip_chunks` predicates are evaluated again. This works like a do-while loop.

### Nesting

A `JobGraph` can be the `job` of a node in another graph. A nested graph counts as
one unit for dependencies: nodes that depend on it wait until the whole nested
graph has finished, including all of its repeats. Labels are namespaced, so error
messages name nodes by their path, like `b.x`.

```julia
step = Ref(0)
inner = JobGraph([
    :forces => (; job=Job(compute_forces!, n)),
    :update => (; job=Job(update_positions!, n), deps=:forces),
]; repeat_if=info -> (step[] += 1) < nsteps)

outer = JobGraph([
    :init     => (; job=Job(initialize!, n)),
    :simulate => (; job=inner, deps=:init),
    :save     => (; do_if=info -> save_output, job=Job(save!, 1), deps=:simulate),
])
```

## Rules for job functions and predicates

- They must be safe to call at the same time as other chunks that are running.
  Chunks of the same job, and of jobs that don't depend on each other, run
  concurrently.
- Per-worker mutable state must be indexed by `info.worker_idx`, not by
  `Threads.threadid()`.
- Each predicate call happens on a single worker, after all of the node's
  dependencies (or all of the graph's nodes, for `repeat_if`) have finished, so it
  sees everything they wrote. Predicates and chunks of unrelated nodes can still run
  at the same time.
- If a function or predicate throws, the other workers stop before starting new
  chunks. `run!` then throws a `JobGraphError` that lists each failing worker's
  exception, backtrace, and location (for example `in job b.x chunk 3`). With one
  worker, `run!` throws an `ErrorException` that names the location, with the
  original exception as its cause.

## Performance notes

- `plan(g, nworkers; compile=true)` precompiles every job function and predicate
  ahead of time.
- With `nworkers == 1`, `run!` simply runs everything in declaration order on the
  calling task.
- A single-chunk job with dependencies, such as a reduction, is run directly by the
  worker that finishes its last dependency.
- Idle workers spin with `yield()` for a while, then sleep until more work is
  published. This frees their threads during a long serial job.
- Callable structs work well as job functions. They avoid boxed captured variables
  and print by type name only.

## Acknowledgments

Thanks to my advisor, Dr. Garegin Papoian, for helpful discussions.

JobGraphs was inspired by:

- [Dagger.jl](https://github.com/JuliaParallel/Dagger.jl)
- [ChunkSplitters.jl](https://github.com/JuliaFolds2/ChunkSplitters.jl)
- [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)
- [Taskflow](https://github.com/taskflow/taskflow)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

## License

MIT, see [LICENSE](LICENSE).
