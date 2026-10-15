module ParallelPlans

# A system for efficient multithreaded programs with mostly static schedules

#=
# Order is priority
foo = StageGraph([
    :a => (;stage=Stage(fun1, 10)),
    :b => (;stage=Stage(fun1_fin, 1), deps=[:a,]),
])

# Nesting is supported (labels are namespaced and become paths)
bar = StageGraph([
    :a => (;stage=Stage(fun2, 10)),
    :b => (;stage=Stage(fun3, 2), deps=[:a,]),
    :c => (;stage=foo, deps=[:a, :b]),
])

# Nodes can have predicates
# StageGraphs can be repeated
bar = StageGraph([
    :a => (;stage=Stage(fun2, 10)),
    :b => (;do_if=info->do_b, stage=Stage(fun3, 2), deps=[:a,]),
    :c => (;skip_chunks=info->8-cld(npairs, pairs_per_chunk), stage=Stage(fun4, 8), deps=[:b,]),
    :d => (;stage=foo, deps=[:a, :b]),
]; repeat_if=info->((step+=1) < N))

# TODO add in needed CUDACore.CuEvents if stages have cuda kernels
# This likely means marking a stage as uses cuda, and padding the CUDACore module to
# plan, so it can use the functions without depending on CUDACore here

my_plan = plan(bar, 16) # 16 workers

run!(my_plan)
=#

"""
    WorkerInfo

Passed to a stage's `fun` and to the `do_if`, `skip_chunks` and `repeat_if`
predicates, describing the worker calling them, see [`Stage`](@ref).

Fields:
- `worker_idx::Int`: the calling worker, in `1:nworkers`. Key per worker
  scratch by it.

One per worker, reused across runs, so don't hold on to it.
"""
mutable struct WorkerInfo
    const worker_idx::Int
end

"""
    Stage(fun, nchunks::Int)

The leaf in a [`StageGraph`](@ref) tree.

`fun(chunk_idx::Int, info::WorkerInfo) -> nothing` is run in parallel across
`nchunks`. `chunk_idx` is in `1:nchunks` and `info` describes the calling
worker, see [`WorkerInfo`](@ref). Capture input and output in `fun`.

A `Stage` with zero chunks is pruned from any [`StageGraph`](@ref) it is put in.

Per worker scratch must be keyed by `info.worker_idx`.
"""
struct Stage
    fun::Any
    nchunks::Int
    Base.@nospecializeinfer function Stage(fun::Any, nchunks::Int)
        @nospecialize
        if nchunks < 0 || nchunks ≥ 2^30
            throw(ArgumentError("nchunks must be in 0:$(2^30 - 1)"))
        end
        new(fun, nchunks)
    end
end

# A do_if predicate normalized to the skip_chunks convention: skip nothing on
# true, everything on false. show unwraps it to print the do_if declaration.
# Concretely typed and called like any other skip_chunks, so the one
# precompile of the wrapper compiles the do_if call inside it.
struct DoIfSkip{F}
    do_if::F
end
(d::DoIfSkip)(info::WorkerInfo) = if d.do_if(info)::Bool
    0
else
    typemax(Int)
end

"""
    StageGraph(nodes::AbstractVector; repeat_if=nothing)

A DAG of [`Stage`](@ref)s and nested `StageGraph`s, declared as a vector of
`label::Symbol => (;skip_chunks, stage, deps)` pairs. `stage` is a `Stage` or a
`StageGraph`; `deps` is a collection of earlier nodes this
node depends on, or a lone `Symbol` for a single dep; `skip_chunks`, or the
simpler `do_if`, is an optional predicate. Declaration order is roughly the
scheduling priority, a load balancing hint.

- `skip_chunks(info::WorkerInfo)::Int` is called once each time the node's
  deps are all done, returning how many of the node's chunks to skip: of the
  node's `nchunks` chunks (1 for a nested `StageGraph`), chunks
  `skip + 1:nchunks` run and the first `skip` don't. So 0 runs the whole node,
  and anything ≥ `nchunks`, `typemax(Int)` say, skips it entirely: nothing in
  it runs, and the nodes that dep on it proceed as if it had run. A negative
  return is an error at runtime.
- `do_if(info::WorkerInfo)::Bool` is the all-or-nothing special case: `true`
  runs the node (a `skip_chunks` of 0), `false` skips it (a `skip_chunks` of
  `typemax(Int)`). A node can have `do_if` or `skip_chunks`, not both.
- `repeat_if(info::WorkerInfo)::Bool` is called each time every node of the
  graph is done. If it returns `true` the graph runs again from the start (its
  `skip_chunks` are reevaluated), else the graph is done, like a do-while loop.

If a `Stage` has zero chunks or a nested `StageGraph` is empty, it will be pruned.
Its `skip_chunks` is never run but transitive dependencies are still respected.
A graph left with no nodes, whether declared empty or emptied by pruning, drops
its `repeat_if`, which is never run.

Build a runnable [`BuiltStageGraph`](@ref) with [`plan`](@ref).
"""
struct StageGraph
    labels::Vector{Symbol}
    stages::Vector{Union{Stage, StageGraph}}
    # Indices of the earlier nodes each node deps on, in decreasing order, with
    # duplicate deps and deps already implied by another dep dropped.
    deps::Vector{Vector{Int}}
    # per node: nothing, or the skip_chunks function, a do_if wrapped in DoIfSkip
    skip_chunks::Vector{Any}
    repeat_if::Any # nothing, or the repeat_if predicate
    Base.@nospecializeinfer function StageGraph(nodes::AbstractVector; repeat_if=nothing)
        @nospecialize repeat_if
        _check_predicate(repeat_if, "repeat_if")
        labels = Symbol[]
        stages = Union{Stage, StageGraph}[]
        deps = Vector{Int}[]
        skip_chunks = Any[]
        # each bit vector has that node and its transitive deps true
        # Used for pruning redundant deps
        all_deps = BitVector[]
        # label => deps of each pruned empty node, as indices into labels, so
        # a dep on a pruned node becomes deps on what it depped on
        pruned_empty = Dict{Symbol, Vector{Int}}()
        for node in nodes
            if !(node isa Pair{Symbol, <:NamedTuple})
                throw(ArgumentError("each node must be a `label::Symbol => (;stage, deps, skip_chunks)` pair, got a $(typeof(node))"))
            end
            label::Symbol = node.first
            props::NamedTuple = node.second
            if label ∈ labels || haskey(pruned_empty, label)
                throw(ArgumentError("duplicate stage label $(repr(label))"))
            end
            if !issubset(propertynames(props), (:stage, :deps, :do_if, :skip_chunks)) || !haskey(props, :stage)
                throw(ArgumentError("stage $(repr(label)): node properties must be `stage` and optionally `deps` and `do_if` or `skip_chunks`, got $(propertynames(props))"))
            end
            stage = props.stage
            if !(stage isa Union{Stage, StageGraph})
                throw(ArgumentError("stage $(repr(label)): `stage` must be a Stage or StageGraph, got a $(typeof(stage))"))
            end
            do_if = get(props, :do_if, nothing)
            _check_predicate(do_if, "stage $(repr(label)): do_if")
            skip_pred = get(props, :skip_chunks, nothing)
            _check_skip_chunks(skip_pred, "stage $(repr(label)): skip_chunks")
            if !isnothing(skip_pred) && !isnothing(do_if)
                throw(ArgumentError("stage $(repr(label)): give `do_if` or `skip_chunks`, not both"))
            end
            if isnothing(skip_pred) && !isnothing(do_if)
                skip_pred = DoIfSkip(do_if) # normalize: downstream it's all skip_chunks
            end
            dep_idxs = Int[]
            raw_deps = get(props, :deps, ())
            if raw_deps isa Symbol
                raw_deps = (raw_deps,)
            end
            for dep in raw_deps
                if !(dep isa Symbol)
                    throw(ArgumentError("stage $(repr(label)): deps must be Symbols, got a $(typeof(dep))"))
                end
                di = findfirst(==(dep), labels)
                if !isnothing(di)
                    push!(dep_idxs, di)
                elseif haskey(pruned_empty, dep)
                    append!(dep_idxs, pruned_empty[dep])
                else
                    throw(ArgumentError("stage $(repr(label)): dep $(repr(dep)) does not name an earlier stage"))
                end
            end
            # Now prune redundant deps. Decreasing order makes one pass exact: a
            # dep can only be implied by a later dep, so by the time a dep is
            # tested every dep that could imply it is already in all_deps_i.
            sort!(dep_idxs; rev=true)
            pruned_dep_idxs = Int[]
            all_deps_i = falses(length(nodes))
            for dep in dep_idxs
                if !all_deps_i[dep]
                    push!(pruned_dep_idxs, dep)
                    all_deps_i .|= all_deps[dep]
                end
            end
            if _is_stage_empty(stage)
                pruned_empty[label] = pruned_dep_idxs
            else
                push!(labels, label)
                push!(stages, stage)
                push!(deps, pruned_dep_idxs)
                push!(skip_chunks, skip_pred)
                all_deps_i[length(labels)] = true
                push!(all_deps, all_deps_i)
            end
        end
        if isempty(labels)
            # Nothing to repeat, whether nodes was empty or all of it pruned.
            repeat_if = nothing
        end
        new(labels, stages, deps, skip_chunks, repeat_if)
    end
end

_is_stage_empty(s::Stage) = iszero(s.nchunks)
_is_stage_empty(s::StageGraph) = isempty(s.labels)

# Predicates can be any callable, but a Bool is a likely mistake, so reject it.
Base.@nospecializeinfer function _check_predicate(pred, what::String)
    @nospecialize pred
    if pred isa Bool
        throw(ArgumentError("$what must be a predicate called as `pred(info::WorkerInfo)`, got the Bool $pred"))
    end
    nothing
end

# skip_chunks can be any callable, but a number is a likely mistake, so reject it.
Base.@nospecializeinfer function _check_skip_chunks(pred, what::String)
    @nospecialize pred
    if pred isa Number
        throw(ArgumentError("$what must be a function called as `skip_chunks(info::WorkerInfo)::Int`, got the number $pred"))
    end
    nothing
end

# Print a fun or predicate: a named function as is, a closure or callable
# struct by its type's name, as its fields may be large.
function _show_callable(io::IO, f)
    if f isa Function && iszero(fieldcount(typeof(f)))
        show(io, f)
    else
        print(io, nameof(typeof(f)))
    end
end

function Base.show(io::IO, s::Stage)
    print(io, Stage, "(")
    _show_callable(io, s.fun)
    print(io, ", ", s.nchunks, ")")
end

# Both show methods print the declaration the graph came from, with the deps the
# constructor dropped left out. The MIME"text/plain" one puts each node on its
# own line, indenting nested graphs.
function Base.show(io::IO, g::StageGraph)
    print(io, StageGraph, "([")
    for i in eachindex(g.labels)
        isone(i) || print(io, ", ")
        _show_node(io, g, i, nothing)
    end
    print(io, "]")
    _show_repeat_if(io, g)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", g::StageGraph)
    _show_multiline(io, g, "")
end

function _show_multiline(io::IO, g::StageGraph, indent::String)
    inner = indent * "    "
    print(io, StageGraph, "([\n")
    for i in eachindex(g.labels)
        print(io, inner)
        _show_node(io, g, i, inner)
        print(io, ",\n")
    end
    print(io, indent, "]")
    _show_repeat_if(io, g)
    print(io, ")")
end

function _show_repeat_if(io::IO, g::StageGraph)
    if !isnothing(g.repeat_if)
        print(io, "; repeat_if=")
        _show_callable(io, g.repeat_if)
    end
end

# Print node i as a `label => (;skip_chunks, stage, deps)` pair, a wrapped
# do_if as the `do_if` it was declared with. `indent` is the
# indentation to print a nested graph at, or nothing to print it on one line.
function _show_node(io::IO, g::StageGraph, i::Int, indent::Union{String, Nothing})
    show(io, g.labels[i])
    print(io, " => (;")
    skip_pred = g.skip_chunks[i]
    if skip_pred isa DoIfSkip
        print(io, "do_if=")
        _show_callable(io, skip_pred.do_if)
        print(io, ", ")
    elseif !isnothing(skip_pred)
        print(io, "skip_chunks=")
        _show_callable(io, skip_pred)
        print(io, ", ")
    end
    print(io, "stage=")
    s = g.stages[i]
    if s isa StageGraph && !isnothing(indent)
        _show_multiline(io, s, indent)
    else
        show(io, s)
    end
    if !isempty(g.deps[i])
        print(io, ", deps=[")
        join(io, (repr(g.labels[d]) for d in g.deps[i]), ", ")
        print(io, "]")
    end
    print(io, ")")
end


# plan turns the declaration into a tree of BuiltGraphNodes, one per graph (a
# StageGraph reused at several nodes gets one per use), whose children are the
# Stages and nested BuiltGraphNodes in declaration order. run! workers scan the
# tree depth first in declaration order, descending into active graphs,
# readying the children that became ready, and claiming chunks of each
# claimable stage they pass until it is exhausted, so declaration order is a
# soft priority. Single chunk stages marked on_fin at build time are never
# scanned: the worker that finishes their last dep chunk runs them itself.
#
# Each BuiltGraphNode owns the runtime counters of its children in one
# AtomicMemory{Int}: a done counter for the graph, then a ready counter and a
# claim counter per child, at a stride of one cache line so children don't
# false-share. The first stride is left empty
# so the done counter doesn't share a line with the memory's header, whose
# pointer every access reads, and the last is left mostly empty so the last
# child's counters don't share one with whatever is allocated next. Counts
# are in chunks, a nested graph counting as one chunk.
#
# The line is 64 bytes on most targets but 128 on Apple silicon
const CACHE_LINE = 128
const STAGE_STATE_STRIDE = CACHE_LINE ÷ sizeof(Int)

# done_ctr: unfinished chunks of the graph's end children, those nothing depend
# on. The worker that takes it to 0 is elected to run repeat_if, then either
# reactivate the graph or finish it as a child of its parent.
_done_idx() = STAGE_STATE_STRIDE + 1
# ready_ctr: unfinished dep chunks of the child. Decremented by the worker that
# finishes each chunk the child deps on, or by a dep's whole chunk count at
# once when that dep is skipped; the one that takes it to 0 marks the child
# ready in its claim_ctr, or runs it itself.
_ready_idx(i::Int) = (i + 1) * STAGE_STATE_STRIDE + 1
# claim_ctr: the child's state, one of these, in the order they happen:
#   CLAIM_NOT_READY    its ready_ctr is not 0 yet; an on_fin stage, run by the
#                      finisher of its last dep chunk, stays here unscanned
#   CLAIM_READY        ready: a stage with a skip_chunks to evaluate, or a
#                      graph.
#                      The first worker to bump it (a stage) or compare-and-swap
#                      it from CLAIM_READY to CLAIM_READYING (a graph) is
#                      elected to ready the child, evaluating its skip_chunks,
#                      if any, and then publishing the stage or activating the
#                      nested graph. A stage with no skip_chunks skips these
#                      two states: whoever readies it publishes it itself. A
#                      graph always goes through them, so its counters are
#                      reset in one place, by the one elected worker, see
#                      _publish!.
#   CLAIM_READYING+k   being readied by a worker
#   0 ≤ n < nchunks    stage: n chunks claimed or skipped so far (a partially
#                      skipped stage is published at its skip count, so bumps
#                      claim chunks skip+1:nchunks); graph: active (n = 0)
#   nchunks ≤ n        exhausted: every chunk claimed (bumps may overshoot),
#                      the child skipped, or the graph done
# Scanning workers read only this to decide what to do with a child. Every
# transition but the bumps is a store, so a bump by a worker that read a
# stale value is harmless: on a not ready, being readied or exhausted count it
# changes nothing that matters, and on CLAIM_READY it is the election, so that
# worker readies the child then. Graphs are elected by a swap, so no bump
# ever lands on an active graph's 0.
_claim_idx(i::Int) = (i + 1) * STAGE_STATE_STRIDE + 2
const CLAIM_READY = 2^61 # > any nchunks plus any overshoot
const CLAIM_READYING = CLAIM_READY + 1
const CLAIM_NOT_READY = 2^62 # > CLAIM_READYING plus any stale bumps

# One graph of a BuiltStageGraph.
mutable struct BuiltGraphNode
    const labels::Vector{Symbol} # per child
    const children::Vector{Union{BuiltGraphNode, Stage}}
    const rev_deps::Vector{Vector{Int}} # per child: the later siblings that dep on it, increasing
    const total_dep_chunks::Vector{Int} # per child: chunks of its deps, the initial ready_ctr
    const total_end_chunks::Int # chunks of the end children, the initial done_ctr
    # per child: nothing, or the skip_chunks function, a do_if wrapped in DoIfSkip
    const skip_chunks::Vector{Any}
    const repeat_if::Any # nothing, or the repeat_if predicate
    # on_fin: per child, whether it is a single chunk stage run by the worker
    # that finishes its last dep chunk, rather than marked ready for scanning.
    # Each child has at most one on_fin rev dep.
    const on_fin::Vector{Bool}
    const scan::Vector{Int} # the children scanning workers look at: those not on_fin
    const states::AtomicMemory{Int} # STAGE_STATE_STRIDE*(length(children)+2) Ints, see _done_idx, _ready_idx and _claim_idx
end

function Base.show(io::IO, node::BuiltGraphNode)
    print(io, BuiltGraphNode, "(", length(node.children), " children)")
end

# Return the BuiltGraphNode for g.
# Its nested graphs are built recursively.
function _build(g::StageGraph, nworkers::Int)
    n = length(g.labels)
    if isone(nworkers) || iszero(n)
        # The dependency tracking isn't needed. Only the root can be empty,
        # empty nested graphs having been pruned, and run! does nothing for it.
        return BuiltGraphNode(
            copy(g.labels), Union{BuiltGraphNode, Stage}[s isa Stage ? s : _build(s, nworkers) for s in g.stages],
            Vector{Int}[], Int[], 0, copy(g.skip_chunks), g.repeat_if,
            Bool[], Int[], AtomicMemory{Int}(undef, 0),
        )
    end
    nchunks = [s isa Stage ? s.nchunks : 1 for s in g.stages]
    rev_deps = [Int[] for _ in 1:n]
    total_dep_chunks = zeros(Int, n)
    for i in 1:n
        for d in g.deps[i]
            push!(rev_deps[d], i) # i increases, so each rev_deps is increasing
            total_dep_chunks[i] += nchunks[d]
        end
    end
    # the last child has no rev deps, so this is not empty
    total_end_chunks = 0
    for i in 1:n
        if isempty(rev_deps[i])
            total_end_chunks += nchunks[i]
        end
    end
    @assert !iszero(total_end_chunks)
    # Greedily mark on_fin stages in declaration order, keeping at most one
    # on_fin rev dep per child.
    on_fin = fill(false, n)
    has_onfin_rev_dep = fill(false, n)
    for i in 1:n
        s = g.stages[i]
        deps = g.deps[i]
        if s isa Stage && isone(s.nchunks) && !isempty(deps) && !any(d -> has_onfin_rev_dep[d], deps)
            on_fin[i] = true
            for d in deps
                has_onfin_rev_dep[d] = true
            end
        end
    end
    states = AtomicMemory{Int}(undef, STAGE_STATE_STRIDE * (n + 2))
    for i in eachindex(states)
        @atomic :monotonic states[i] = CLAIM_NOT_READY # nothing to do until activated
    end
    BuiltGraphNode(
        copy(g.labels), Union{BuiltGraphNode, Stage}[s isa Stage ? s : _build(s, nworkers) for s in g.stages],
        rev_deps, total_dep_chunks, total_end_chunks, copy(g.skip_chunks), g.repeat_if,
        on_fin, findall(!, on_fin), states,
    )
end

"""
    plan(g::StageGraph, nworkers::Int; compile=true)::BuiltStageGraph

Plan `g` for [`run!`](@ref) on `nworkers` workers.

With more than one worker and `compile`, every `fun` and predicate of `g` is
compiled here, so the first `run!` does not compile them in a worker while
the others wait. That is a compile per new type of `fun` or predicate, and a
lookup of about a microsecond per stage once they are compiled, so pass
`compile=false` when planning often.

The returned object cannot be `run!` concurrently.
"""
function plan(g::StageGraph, nworkers::Int; compile::Bool=true)
    if nworkers ≤ 0
        throw(ArgumentError("nworkers must be positive"))
    end
    if nworkers ≥ 2^16
        throw(ArgumentError("nworkers must be less than 2^16. Got $(nworkers)"))
    end
    built = BuiltStageGraph(_build(g, nworkers), nworkers)
    if compile && nworkers > 1
        _compile!(built.root)
    end
    built
end

# Compile what the parallel run! calls for each fun and predicate of the tree
# under node: the predicates; the funs of on_fin stages, which _finish!
# calls directly; and the claim loop specialized on the fun of every scanned stage,
# which calls it statically inside.
function _compile!(node::BuiltGraphNode)
    for (i, child) in enumerate(node.children)
        skip_pred = node.skip_chunks[i]
        isnothing(skip_pred) || precompile(skip_pred, (WorkerInfo,))
        if child isa BuiltGraphNode
            _compile!(child)
        elseif node.on_fin[i]
            precompile(child.fun, (Int, WorkerInfo))
        else
            precompile(_claim!, (typeof(child.fun), BuiltStageGraph, WorkerState, BuiltGraphNode, Int, Int))
        end
    end
    isnothing(node.repeat_if) || precompile(node.repeat_if, (WorkerInfo,))
    nothing
end

struct StageGraphWorkerError
    worker_idx::Int
    label::Vector{Symbol} # node whose fun or predicate threw; empty for the root graph
    chunk_idx::Int # > 0 chunk of the stage; 0 the node's skip_chunks; -1 the graph's repeat_if; 0 with an empty label: outside any fun or predicate
    err::CapturedException
end

# Thrown by run! when workers error; shows each worker's exception and backtrace.
struct StageGraphError <: Exception
    worker_errors::Vector{StageGraphWorkerError}
end

function Base.showerror(io::IO, e::StageGraphError)
    print(io, "StageGraphError: ", length(e.worker_errors), " worker(s) errored")
    for we in e.worker_errors
        print(io, "\n\nworker ", we.worker_idx, " ", _where_string(we.label, we.chunk_idx), ": ")
        showerror(io, we.err)
    end
end

# Return where a worker was when it threw, like "in stage a.b chunk 3".
# From its attribution fields, see StageGraphWorkerError.
function _where_string(label::Vector{Symbol}, chunk_idx::Int)
    if isempty(label) && iszero(chunk_idx)
        return "outside any stage or predicate"
    end
    name = isempty(label) ? "the root graph" : join(label, '.')
    if chunk_idx > 0
        "in stage $name chunk $chunk_idx"
    elseif iszero(chunk_idx)
        "in the skip_chunks of $name"
    else
        "in the repeat_if of $name"
    end
end

# Per worker scratch, owned by one worker for the length of a run!.
mutable struct WorkerState
    const info::WorkerInfo # passed to the funs and predicates this worker calls
    # The fun or predicate running, for error attribution, as (graph, child,
    # chunk_idx as StageGraphWorkerError.chunk_idx); child 0 is the graph's
    # repeat_if, and (root, 0, 0) is none.
    at::Tuple{BuiltGraphNode, Int, Int}
    err::Union{Nothing, StageGraphWorkerError}
    WorkerState(idx::Int, root::BuiltGraphNode) = new(WorkerInfo(idx), (root, 0, 0), nothing)
end

# Return the label path of what w.at attributes an error to, under root.
# The tree is searched for w.at. Errors are rare, so this beats
# keeping a path up to date in every scan.
function _at_label(w::WorkerState, root::BuiltGraphNode)
    (node, i, _) = w.at
    label = Symbol[]
    found = _find_graph!(label, root, node)
    @assert found
    iszero(i) || push!(label, node.labels[i])
    label
end
function _find_graph!(label::Vector{Symbol}, node::BuiltGraphNode, target::BuiltGraphNode)
    node === target && return true
    for (i, child) in enumerate(node.children)
        child isa BuiltGraphNode || continue
        push!(label, node.labels[i])
        _find_graph!(label, child, target) && return true
        pop!(label)
    end
    false
end

# For a function barrier on fun, used by the serial run.
function _run_chunks!(fun, w::WorkerState, node::BuiltGraphNode, i::Int, skip::Int, nchunks::Int)
    for chunk_idx in (skip + 1):nchunks
        w.at = (node, i, chunk_idx)
        fun(chunk_idx, w.info)
    end
    nothing
end

# Return how many of node child i's chunks its skip_chunks skips, ≥ 0 but not
# capped at the chunk count: 0 without one. A do_if skips nothing on true and
# everything on false.
@inline function _call_skip_chunks(w::WorkerState, node::BuiltGraphNode, i::Int)::Int
    skip_pred = node.skip_chunks[i]
    isnothing(skip_pred) && return 0
    w.at = (node, i, 0)
    skip = skip_pred(w.info)::Int
    skip < 0 && _throw_negative_skip(skip)
    skip
end
@noinline _throw_negative_skip(skip::Int) =
    throw(ArgumentError("skip_chunks returned $skip, must be ≥ 0"))

# Return whether to repeat node based on repeat_if.
function _call_repeat_if(w::WorkerState, node::BuiltGraphNode)::Bool
    isnothing(node.repeat_if) && return false
    w.at = (node, 0, -1)
    node.repeat_if(w.info)::Bool
end

# Padding to keep an atomic field on its own cache line in BuiltStageGraph.
const CACHE_LINE_PAD = Tuple(zeros(UInt8, CACHE_LINE - sizeof(UInt64)))

"""
    BuiltStageGraph

A [`StageGraph`](@ref) instantiated by [`plan`](@ref) for a fixed number of
workers, ready for [`run!`](@ref). Holds the tree of graphs with their runtime
counters, and the per worker scratch `run!` uses, so one `BuiltStageGraph` can
be run many times but not concurrently. An empty graph has `nworkers == 0` and
no worker scratch.
"""
mutable struct BuiltStageGraph
    const root::BuiltGraphNode
    const nworkers::Int
    const workers::Vector{WorkerState}
    # set by the worker that finishes the root graph, or errors; checked before starting a chunk
    @atomic exit::Bool
    const sleepers::Threads.Condition
    # Padding on both sides keeps notify_sleeper on its own cache line.
    const _pad_before::typeof(CACHE_LINE_PAD)
    # 16 lsb are a count of workers going to sleep or sleeping, 48 msb are the notify counter
    @atomic notify_sleeper::UInt64
    const _pad_after::typeof(CACHE_LINE_PAD)
    function BuiltStageGraph(root::BuiltGraphNode, nworkers::Int)
        @assert nworkers > 0
        if isempty(root.children)
            nworkers = 0 # an empty graph has nothing to run, so no workers
        end
        new(root, nworkers, [WorkerState(i, root) for i in 1:nworkers], false, Threads.Condition(), CACHE_LINE_PAD, UInt64(0), CACHE_LINE_PAD)
    end
end

function Base.show(io::IO, g::BuiltStageGraph)
    print(io, BuiltStageGraph, "(@", repr(objectid(g)), ", ", g.nworkers, " workers)")
end

"""
    run!(g::BuiltStageGraph)

Run every chunk of every stage as `stage.fun(chunk_idx, info::WorkerInfo)` on
`g.nworkers` workers, honoring the `skip_chunks`, and `repeat_if`
predicates. No chunk
of a stage starts until every chunk of each stage it deps on, directly or
transitively, has finished, so a chunk can read whatever those chunks produced
without synchronizing in the `fun`s. A node that deps on a nested graph waits
for the whole graph, repeats included. Independent ready stages overlap, with
chunks claimed in roughly declaration order. With one worker everything runs
in declaration order on the calling task, with no spawns or counters.

A worker with no chunks to work on `yield()` spins, and after a while sleeps
until another worker makes chunks claimable that it could not take itself,
so idle workers free their threads during a long stage.

`fun`s and predicates must be safe to call concurrently with other in flight
chunks. Per worker scratch must be keyed by `info.worker_idx`. If a `fun` or
predicate throws the other workers wind down and
`run!` throws a `StageGraphError` collecting every worker's error. With one
worker it throws an `ErrorException` naming the stage and chunk, with the
`fun`'s exception as its cause.

Not reentrant: the counters live in `g`, so don't call `run!` on the same
`BuiltStageGraph` concurrently.
"""
function run!(g::BuiltStageGraph)
    if g.nworkers == 0
        return # an empty graph
    end
    if g.nworkers == 1
        return _run_serial!(g)
    end
    # exit is monotonic everywhere: the flag carries no data. The error records
    # in the worker states are synchronized by the @sync task join, and these
    # resets are published to the workers by the @spawn itself.
    @atomic :monotonic g.exit = false
    @atomic :monotonic g.notify_sleeper = UInt64(0)
    for w in g.workers
        w.err = nothing
    end
    _activate!(g.root)
    @sync begin
        for w in g.workers
            Threads.@spawn _run_worker(g, $w)
        end
    end
    errs = StageGraphWorkerError[w.err for w in g.workers if !isnothing(w.err)]
    isempty(errs) || throw(StageGraphError(errs))
    nothing
end

# Single worker: interpret the tree in declaration order on the calling task.
# This is also the reference semantics for the parallel run.
function _run_serial!(g::BuiltStageGraph)
    w = g.workers[1]
    w.at = (g.root, 0, 0)
    try
        _run_serial_graph!(w, g.root)
    catch
        error("StageGraph run! errored ", _where_string(_at_label(w, g.root), w.at[3]))
    end
    nothing
end

# Run graph node. Recursion depth is the nesting depth of the declaration;
# repeats loop.
function _run_serial_graph!(w::WorkerState, node::BuiltGraphNode)
    while true
        for (i, child) in enumerate(node.children)
            skip = _call_skip_chunks(w, node, i)
            nchunks = child isa Stage ? child.nchunks : 1 # a nested graph is one chunk
            if skip < nchunks
                if child isa Stage
                    _run_chunks!(child.fun, w, node, i, skip, child.nchunks)
                elseif child isa BuiltGraphNode
                    _run_serial_graph!(w, child)
                else
                    error("unreachable")
                end
            end
        end
        _call_repeat_if(w, node) || break
    end
    nothing
end

const MAX_YIELD_SPINS = 1000 # number of yield spins before the worker goes to sleep
const ONE_NOTIFY_GEN = UInt64(1)<<16

# Set the exit flag and wake the sleepers.
function _mark_exit!(g::BuiltStageGraph)
    @atomic g.exit = true
    @lock g.sleepers notify(g.sleepers)
end
# Wake the sleeping workers, if any, after the caller published more work than it can do itself.
#
# A worker going to sleep bumps the sleeper count, does a final scan, and
# sleeps only if that scan found nothing and the generation is still what its
# bump returned. Both bumps are RMWs on notify_sleeper, so one comes first in
# its modification order. If this one does, the sleeper's bump acquires it,
# so the sleeper's final scan sees the publish. If the sleeper's does, the
# count returned here includes it, and the notify below either changes the
# generation before the sleeper checks it or wakes the sleeper.
@inline function _wake_sleeping!(g::BuiltStageGraph)
    x = @atomic g.notify_sleeper += ONE_NOTIFY_GEN
    if !iszero(x%UInt16)
        _do_notify!(g)
    end
end
@noinline function _do_notify!(g::BuiltStageGraph)
    @lock g.sleepers notify(g.sleepers)
end

# A worker scans the tree until the root is done or a worker errors, yielding
# after a scan that ran nothing.
function _run_worker(g::BuiltStageGraph, w::WorkerState)
    w.at = (g.root, 0, 0)
    try
        yield_count = 0
        final_scan = false
        while !(@atomic :monotonic g.exit)
            # Check if this is the final scan before going to sleep
            if final_scan
                # Sign in as a sleeper before the final scan, see _wake_sleeping!.
                start_notify_sleeper = @atomic g.notify_sleeper += UInt64(1)
                start_notify_gen = start_notify_sleeper >> 16
            end
            # TODO tryout a first pass fresh only flag to spread out workers more, to improve performance
            scanned = _scan!(g, w, g.root)
            if scanned.done
                _mark_exit!(g) # the root is done
                return
            end
            if !final_scan
                if scanned.progress
                    yield_count = 0
                else
                    yield_count += 1
                    yield()
                    final_scan = yield_count ≥ MAX_YIELD_SPINS
                end
            else
                if !scanned.progress
                    # sleep
                    @lock g.sleepers begin
                        while ((@atomic g.notify_sleeper)>>16 == start_notify_gen) && !(@atomic g.exit)
                            wait(g.sleepers)
                        end
                    end
                end
                @atomic g.notify_sleeper -= UInt64(1)
                yield_count = 0
                final_scan = false
                (@atomic g.exit) && return
            end
        end
    catch e
        _mark_exit!(g)
        # w.at attributes an error to its stage and chunk or predicate; the
        # label distinguishes nodes that share the same fun. Each worker owns
        # its state. The other workers see the flag before their next chunk and
        # wind down without being interrupted: nothing can dep on the throwing
        # chunk.
        w.err = StageGraphWorkerError(w.info.worker_idx, _at_label(w, g.root), w.at[3], CapturedException(e, catch_backtrace()))
    end
    nothing
end

# Return (; progress, done).
# `progress` is true if anything was worked on.
# `done` is true if the node is done.
# Scan the children of graph node in declaration order, readying the ready
# ones, claiming chunks of the claimable stages and descending into the active
# graphs.
function _scan!(g::BuiltStageGraph, w::WorkerState, node::BuiltGraphNode)
    states = node.states
    progress = false
    for i in node.scan
        (@atomic :monotonic g.exit) && return (progress=true, done=false)
        child = node.children[i]
        if child isa Stage
            # A plain load only decides what to try; the RMWs below are what
            # actually gate, so a stale value is harmless.
            claimed = @inbounds @atomic :monotonic states[_claim_idx(i)]
            if !(claimed < child.nchunks || claimed == CLAIM_READY)
                continue
            end
            # try to do work on the stage.
            # ndone is the number of chunks completed
            ndone = _claim!(child.fun, g, w, node, i, child.nchunks)
            iszero(ndone) && continue
            progress = true
            done = _finish!(g, w, node, i, ndone)
            done && return (progress=true, done=true)
        elseif child isa BuiltGraphNode
            # We need acquire to ensure we don't descend into the subgraph
            # before the _activate!(child) call has initialized it.
            claimed = @inbounds @atomic :acquire states[_claim_idx(i)]
            if !(iszero(claimed) || claimed == CLAIM_READY)
                continue
            end
            if claimed == CLAIM_READY
                # The acquire pairs with the release store of CLAIM_READY, so
                # the dep chunks' writes are visible to skip_chunks and the graph.
                (; success) = @inbounds @atomicreplace :acquire :monotonic states[_claim_idx(i)] CLAIM_READY => CLAIM_READYING
                success || continue
                progress = true
                # elected to ready child i: skip it, or activate it and scan it
                if _call_skip_chunks(w, node, i) > 0 # skip the one chunk a graph is: finish it at once, as if it had run
                    @inbounds @atomic :monotonic states[_claim_idx(i)] = 1 # exhausted
                    done = _finish!(g, w, node, i, 1)
                    done && return (progress=true, done=true)
                    continue
                end
                npublished = _activate!(child)
                # Swap rather than store: a worker that read a stale 0 here
                # from the child's last activation can descend now, run what
                # _activate! just published, finish the child and store 1
                # before this line. A store of 0 would then mark a done graph
                # active.
                # Release pairs with the acquire above.
                @inbounds @atomicreplace :release :monotonic states[_claim_idx(i)] CLAIM_READYING => 0 # active
                # Only now can other workers descend into child. This worker
                # takes one of the published chunks below.
                npublished > 1 && _wake_sleeping!(g)
            end
            scanned = _scan!(g, w, child)
            progress |= scanned.progress
            if scanned.done # finish it here, as child i
                @inbounds @atomic :monotonic states[_claim_idx(i)] = 1 # scanners stop descending
                done = _finish!(g, w, node, i, 1)
                done && return (progress=true, done=true)
            end
        else
            error("unreachable")
        end
    end
    (; progress, done=false)
end

# Return the number of chunks completed or skipped.
# Claim and run child i's chunks, until it is exhausted, readying
# it first if elected to: its skip_chunks, if any, picks how many chunks to
# skip, and the claim counter starts there, so chunks skip+1:nchunks are
# claimed as usual and the skipped ones finish with this worker's batch.
# Function barrier on fun, so the loop dispatches on fun's type once rather
# than at every chunk
function _claim!(fun, g::BuiltStageGraph, w::WorkerState, node::BuiltGraphNode, i::Int, nchunks::Int)
    states = node.states
    ndone = 0
    while !(@atomic :monotonic g.exit)
        # The acquire pairs with the release store of CLAIM_READY or the skip
        # count by the worker that readied the stage,
        # so the dep chunks' writes are visible before skip_chunks or fun runs.
        chunk_idx = @inbounds @atomic :acquire states[_claim_idx(i)] += 1
        if chunk_idx == CLAIM_READYING # it was CLAIM_READY: elected to ready it
            skip = min(_call_skip_chunks(w, node, i), nchunks)
            if skip == nchunks
                @inbounds @atomic :monotonic states[_claim_idx(i)] = nchunks # exhausted
                return nchunks
            end
            ndone += skip # this worker finishes the skipped chunks
            @inbounds @atomic :release states[_claim_idx(i)] = skip # chunks skip+1:nchunks claimable
            # This worker takes one chunk below.
            nchunks - skip > 1 && _wake_sleeping!(g)
        elseif chunk_idx ≤ nchunks
            w.at = (node, i, chunk_idx)
            fun(chunk_idx, w.info)
            ndone += 1
        else
            break # exhausted
        end
    end
    ndone
end

# Initialize the counters of node's children for a new activation and mark its
# starts, the children with no deps, ready. Only the activating worker touches
# the counters until then. Nested graphs are not activated here: a graph start
# is marked ready for the scan to elect a worker to activate it, see _publish!.
function _activate!(node::BuiltGraphNode)
    states = node.states
    for i in eachindex(node.children)
        @inbounds @atomic :monotonic states[_ready_idx(i)] = node.total_dep_chunks[i]
        @inbounds @atomic :monotonic states[_claim_idx(i)] = CLAIM_NOT_READY
    end
    @inbounds @atomic :monotonic states[_done_idx()] = node.total_end_chunks
    npublished = 0
    for i in eachindex(node.children)
        if iszero(node.total_dep_chunks[i])
            npublished += _publish!(node, i)
        end
    end
    npublished
end

# Return the number of work items made available to workers
# Mark child i of graph node ready, its deps being done: a graph, or a stage
# with a skip_chunks, for the scan to elect a worker to ready it; a stage
# without, claimable at once.
function _publish!(node::BuiltGraphNode, i::Int)::Int
    states = node.states
    child = node.children[i]
    if child isa Stage && isnothing(node.skip_chunks[i])
        @inbounds @atomic :release states[_claim_idx(i)] = 0 # claimable
        child.nchunks
    else
        @inbounds @atomic :release states[_claim_idx(i)] = CLAIM_READY
        1
    end
end

# Return whether graph node is done and not repeating.
# May return false if another worker errored and set g.exit.
function _finish!(g::BuiltStageGraph, w::WorkerState, node::BuiltGraphNode, i::Int, n::Int)::Bool
    states = node.states
    # Loop along the chain of on_fin stages
    while true
        rev_deps = node.rev_deps[i]
        if isempty(rev_deps) # nothing depends on me
            # check if this finishes the node
            # A negative count means a chunk was finished twice.
            left = @inbounds @atomic :acquire_release states[_done_idx()] -= n
            @assert left ≥ 0
            if !iszero(left)
                return false
            end
            # Try to early exit before call to repeat_if
            (@atomic :monotonic g.exit) && return false
            # every child of node is done and visible to this worker
            if _call_repeat_if(w, node)
                # node stays active in its parent, so the starts are claimable
                # at once. This worker takes one chunk in its next scan.
                npublished = _activate!(node)
                npublished > 1 && _wake_sleeping!(g)
                return false
            else
                return true
            end
        end
        on_fin = 0
        npublished = 0
        for r in rev_deps
            # Release publishes everything this worker has seen of child i and
            # its deps; acquire gets the same from the other deps' finishers.
            left = @inbounds @atomic :acquire_release states[_ready_idx(r)] -= n
            @assert left ≥ 0 # a double finish would step over 0
            if !iszero(left)
                continue
            end
            if node.on_fin[r]
                # Run below by this worker
                @assert iszero(on_fin)
                on_fin = r
            else
                npublished += _publish!(node, r)
            end
        end
        # This worker takes one chunk itself: the on_fin stage, run below, or
        # else a published chunk in its scan. Wake before running the on_fin
        # stage, so sleepers don't wait it out.
        if npublished + !iszero(on_fin) > 1
            _wake_sleeping!(g)
        end
        iszero(on_fin) && return false
        (@atomic :monotonic g.exit) && return false
        # an on_fin stage has one chunk, so any positive skip skips it
        if iszero(_call_skip_chunks(w, node, on_fin))
            w.at = (node, on_fin, 1)
            (@atomic :monotonic g.exit) && return false
            (node.children[on_fin]::Stage).fun(1, w.info)
        end
        # now finish the on_fin stage
        i, n = on_fin, 1
    end
end

end # module ParallelPlans
