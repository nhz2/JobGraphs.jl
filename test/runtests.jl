using ParallelPlans
using Test
using Random
using Aqua: Aqua

Aqua.test_all(ParallelPlans)

# Job funs and predicates the construction and plan tests never call.
job_fun_a(chunk_idx, info) = nothing
job_fun_b(chunk_idx, info) = nothing
pred_a(info) = true
pred_b(info) = false
skip_a(info) = 1

# Recording runs. A job fun records each chunk it runs with ticks of a
# shared clock taken before and after its work, so intervals compare across
# workers, and a predicate counts its calls.

struct RunEvent
    id::Symbol
    chunk::Int
    worker::Int
    t0::Int
    t1::Int
end

mutable struct RunCtx
    const lock::ReentrantLock
    const events::Vector{RunEvent}
    const calls::Dict{Symbol, Int} # per predicate id
    nworkers::Int
    @atomic clock::Int
end
RunCtx() = RunCtx(ReentrantLock(), RunEvent[], Dict{Symbol, Int}(), 0, 0)

function run_reset!(ctx::RunCtx, nworkers::Int)
    empty!(ctx.events)
    empty!(ctx.calls)
    ctx.nworkers = nworkers
    @atomic ctx.clock = 0
    ctx
end

# Throw unless `info` is what a fun or predicate is passed, see WorkerInfo.
function check_info(ctx::RunCtx, info)
    info isa ParallelPlans.WorkerInfo || error("expected a WorkerInfo, got a $(typeof(info))")
    info.worker_idx ∈ 1:ctx.nworkers || error("bad worker_idx $(info.worker_idx) with $(ctx.nworkers) workers")
    nothing
end

# A job fun that records its chunks as `id`.
struct RunFun
    ctx::RunCtx
    id::Symbol
end
function (f::RunFun)(chunk_idx, info)
    ctx = f.ctx
    check_info(ctx, info)
    t0 = @atomic ctx.clock += 1
    x = 0.0
    for i in 1:500
        x += sin(i)
    end
    t1 = @atomic ctx.clock += 1
    lock(ctx.lock) do
        push!(ctx.events, RunEvent(f.id, chunk_idx, info.worker_idx, t0, t1))
    end
    x
end

# A job fun that sleeps `secs` then records its chunk as `f` does, so
# the other workers run out of spins and sleep meanwhile.
struct SlowFun
    f::RunFun
    secs::Float64
end
function (f::SlowFun)(chunk_idx, info)
    sleep(f.secs)
    f.f(chunk_idx, info)
end

# A predicate that counts its calls as `id` and returns f(ncalls).
struct RunPred
    ctx::RunCtx
    id::Symbol
    f::Any
end
function (p::RunPred)(info)
    ctx = p.ctx
    check_info(ctx, info)
    n = lock(ctx.lock) do
        ctx.calls[p.id] = get(ctx.calls, p.id, 0) + 1
    end
    p.f(n)::Bool
end

# A skip_chunks function that counts its calls as `id` and returns f(ncalls).
struct RunSkip
    ctx::RunCtx
    id::Symbol
    f::Any
end
function (p::RunSkip)(info)
    ctx = p.ctx
    check_info(ctx, info)
    n = lock(ctx.lock) do
        ctx.calls[p.id] = get(ctx.calls, p.id, 0) + 1
    end
    p.f(n)::Int
end

# Whether no two events on the same worker overlap. Events have `worker`,
# `t0` and `t1` fields.
function workers_serial(events::Vector)
    for i in eachindex(events), j in 1:i-1
        a = events[i]
        b = events[j]
        if a.worker == b.worker && a.t0 < b.t1 && b.t0 < a.t1
            return false
        end
    end
    true
end

# Run plan b on a reset ctx and return the (id, chunk) pairs recorded, sorted,
# checking no worker overlapped its own chunks.
function run_chunks(ctx::RunCtx, b::ParallelPlans.BuiltJobGraph)
    run_reset!(ctx, b.nworkers)
    @test isnothing(ParallelPlans.run!(b))
    @test workers_serial(ctx.events)
    sort!([(e.id, e.chunk) for e in ctx.events])
end
run_chunks(ctx::RunCtx, g::ParallelPlans.JobGraph, nw::Int) = run_chunks(ctx, ParallelPlans.plan(g, nw))

# The ids of the last run's chunks in start order.
run_order(ctx::RunCtx) = [e.id for e in sort(ctx.events; by=e -> e.t0)]

# Whether in the last run every chunk with an id in `before` ended before any
# chunk with an id in `after` started.
function run_before(ctx::RunCtx, before::Vector{Symbol}, after::Vector{Symbol})
    maximum(e.t1 for e in ctx.events if e.id ∈ before) < minimum(e.t0 for e in ctx.events if e.id ∈ after)
end

# The exception run! throws for plan b, or nothing. Resets ctx.
function run_error(ctx::RunCtx, b::ParallelPlans.BuiltJobGraph)
    run_reset!(ctx, b.nworkers)
    try
        ParallelPlans.run!(b)
        nothing
    catch e
        e
    end
end
run_error(ctx::RunCtx, g::ParallelPlans.JobGraph, nw::Int) = run_error(ctx, ParallelPlans.plan(g, nw))

# Where a parallel run's error says each worker was, like "in job a chunk 1".
# Through the same function as showerror, which is not called on real errors
# here as printing backtraces is slow to compile.
error_wheres(e::ParallelPlans.JobGraphError) = [ParallelPlans._where_string(we.label, we.chunk_idx) for we in e.worker_errors]

# A predicate of g throws: the error is attributed to it, serially and on NW workers.
function check_pred_error(ctx::RunCtx, g::ParallelPlans.JobGraph, where::String, NW::Int)
    e = run_error(ctx, g, 1)
    @test e isa ErrorException
    @test occursin(where, e.msg)
    e = run_error(ctx, g, NW)
    @test e isa ParallelPlans.JobGraphError
    @test error_wheres(e) == [where]
end

# Check a plan for one worker against one for several from the same
# declaration: the same tree, with the dep tracking, on_fin marking and
# counters left out.
function check_serial_plan(node::ParallelPlans.BuiltGraphNode, ref::ParallelPlans.BuiltGraphNode)
    @test node.labels == ref.labels
    @test node.skip_chunks == ref.skip_chunks
    @test node.repeat_if === ref.repeat_if
    @test isempty(node.rev_deps) && isempty(node.total_dep_chunks) && iszero(node.total_end_chunks)
    @test isempty(node.on_fin) && isempty(node.scan) && isempty(node.states)
    @test length(node.children) == length(ref.children)
    for (c, r) in zip(node.children, ref.children)
        if c isa ParallelPlans.Job
            @test c === r
        else
            @test r isa ParallelPlans.BuiltGraphNode
            check_serial_plan(c, r)
        end
    end
end

# Random nested graphs, run serially for a reference and then in parallel.
# Nodes have integer ids. Each chunk records the key of its activation: the
# completed iteration counts of the graphs on its path, root first, which
# their repeat_ifs bump. Chunks with the same key ran in the same activation
# of the same graph.

struct XEvent
    id::Int
    chunk::Int
    key::Vector{Int}
    worker::Int
    t0::Int
    t1::Int
end

mutable struct XCtx
    const lock::ReentrantLock
    const events::Vector{XEvent}
    const iter::Vector{Int} # per graph id: completed iterations
    const calls::Vector{Int} # per node id: do_if calls
    nworkers::Int
    @atomic clock::Int
end

# The structure of a random graph, by node id.
struct XInfo
    deps::Vector{Vector{Int}} # sibling ids, through pruned empty siblings
    ancestors::Vector{Vector{Int}} # graph ids from the root down to the parent
    nrepeat::Vector{Int} # graphs: iterations per activation; jobs: 1
end

function x_reset!(ctx::XCtx, nworkers::Int)
    empty!(ctx.events)
    fill!(ctx.iter, 0)
    fill!(ctx.calls, 0)
    ctx.nworkers = nworkers
    @atomic ctx.clock = 0
    ctx
end

struct XFun
    ctx::XCtx
    id::Int
    ancestors::Vector{Int}
end
function (f::XFun)(chunk_idx, info)
    ctx = f.ctx
    info.worker_idx ∈ 1:ctx.nworkers || error("bad worker_idx $(info.worker_idx)")
    key = [ctx.iter[a] for a in f.ancestors]
    t0 = @atomic ctx.clock += 1
    x = 0.0
    for i in 1:100
        x += sin(i)
    end
    t1 = @atomic ctx.clock += 1
    lock(ctx.lock) do
        push!(ctx.events, XEvent(f.id, chunk_idx, key, info.worker_idx, t0, t1))
    end
    x
end

# kind 1: always run; 2: every other time; 3: never
struct XDoIf
    ctx::XCtx
    id::Int
    kind::Int
end
function (p::XDoIf)(info)
    info.worker_idx ∈ 1:p.ctx.nworkers || error("bad worker_idx $(info.worker_idx)")
    n = (p.ctx.calls[p.id] += 1) # nodes are readied one at a time, so no lock needed
    p.kind == 1 || (p.kind == 2 && isodd(n))
end

# A skip_chunks cycling through 0:5 per call: none, partial or all of the
# 0:4 chunks a random node has.
struct XSkipChunks
    ctx::XCtx
    id::Int
end
function (p::XSkipChunks)(info)
    info.worker_idx ∈ 1:p.ctx.nworkers || error("bad worker_idx $(info.worker_idx)")
    n = (p.ctx.calls[p.id] += 1) # nodes are readied one at a time, so no lock needed
    (5n + p.id) % 6 # 5 is coprime with 6, so calls cycle through every count
end

# Exactly nrepeat iterations per activation.
struct XRepeatIf
    ctx::XCtx
    id::Int
    nrepeat::Int
end
function (p::XRepeatIf)(info)
    info.worker_idx ∈ 1:p.ctx.nworkers || error("bad worker_idx $(info.worker_idx)")
    n = (p.ctx.iter[p.id] += 1)
    n % p.nrepeat != 0
end

# Register a node and return its id.
function x_new!(info::XInfo, ancestors::Vector{Int})
    push!(info.deps, Int[])
    push!(info.ancestors, ancestors)
    push!(info.nrepeat, 1)
    length(info.deps)
end

# The declaration of graph `id`, registered already, nested at most `depth`
# deep with 0:5 nodes per level (1:5 at the root), random deps between
# siblings, random predicates and 1:3 iterations per activation. Jobs have
# 0:4 chunks, so some nodes are empty and pruned: a dep on one is recorded as
# deps on its deps.
function x_rand_graph!(rng, ctx::XCtx, info::XInfo, id::Int, ancestors::Vector{Int}, depth::Int)
    anc = vcat(ancestors, id)
    nodes = Pair{Symbol, NamedTuple}[]
    cids = Int[]
    empties = Bool[]
    for i in 1:rand(rng, (isempty(ancestors) ? 1 : 0):5)
        cid = x_new!(info, anc)
        deps = Symbol[]
        for j in 1:i-1
            rand(rng) < 0.4 || continue
            push!(deps, Symbol(:n, j))
            if empties[j]
                append!(info.deps[cid], info.deps[cids[j]])
            else
                push!(info.deps[cid], cids[j])
            end
        end
        job = if depth > 0 && rand(rng) < 0.3
            x_rand_graph!(rng, ctx, info, cid, anc, depth - 1)
        else
            ParallelPlans.Job(XFun(ctx, cid, anc), rand(rng, 0:4))
        end
        kind = rand(rng, 0:4)
        props = if iszero(kind)
            (; job, deps)
        elseif kind == 4
            (; skip_chunks=XSkipChunks(ctx, cid), job, deps)
        else
            (; do_if=XDoIf(ctx, cid, kind), job, deps)
        end
        push!(nodes, Symbol(:n, i) => props)
        push!(cids, cid)
        push!(empties, ParallelPlans._is_job_empty(job))
    end
    nrepeat = rand(rng, 1:3)
    info.nrepeat[id] = nrepeat
    repeat_if = isone(nrepeat) && rand(rng, Bool) ? nothing : XRepeatIf(ctx, id, nrepeat)
    ParallelPlans.JobGraph(nodes; repeat_if)
end

# A random root graph with its ctx and info.
function x_rand_graph(rng, depth::Int)
    ctx = XCtx(ReentrantLock(), XEvent[], Int[], Int[], 0, 0)
    info = XInfo(Vector{Int}[], Vector{Int}[], Int[])
    root = x_new!(info, Int[])
    g = x_rand_graph!(rng, ctx, info, root, Int[], depth)
    resize!(ctx.iter, length(info.deps))
    resize!(ctx.calls, length(info.deps))
    (g, ctx, info)
end

# Sorted (id, chunk, key) of every event.
x_chunks(ctx::XCtx) = sort!([(e.id, e.chunk, e.key) for e in ctx.events])

# Whether the last run's events respect the declaration: within an activation
# of a graph a node's chunks (a nested graph counting as all of its chunks)
# end before a dependent's start, the iterations of a graph don't overlap,
# and a worker runs one chunk at a time.
function x_ordered(info::XInfo, ctx::XCtx)
    # (graph, activation key, child) => interval of the child's subtree, and
    # with child 0, of the whole activation
    ivals = Dict{Tuple{Int, Vector{Int}, Int}, Tuple{Int, Int}}()
    for e in ctx.events
        anc = info.ancestors[e.id]
        for d in eachindex(anc), child in (d < length(anc) ? anc[d+1] : e.id, 0)
            k = (anc[d], e.key[1:d], child)
            (lo, hi) = get(ivals, k, (typemax(Int), 0))
            ivals[k] = (min(lo, e.t0), max(hi, e.t1))
        end
    end
    ok = true
    for ((p, key, y), (ylo, _)) in ivals
        iszero(y) && continue
        for x in info.deps[y]
            xi = get(ivals, (p, key, x), nothing)
            isnothing(xi) || (ok &= xi[2] < ylo)
        end
    end
    # two iterations of one graph in the same activation of its parent
    iters = [(p, key, lo, hi) for ((p, key, child), (lo, hi)) in ivals if iszero(child)]
    for (p, key, lo, hi) in iters, (q, key2, lo2, hi2) in iters
        if p == q && key != key2 && key[1:end-1] == key2[1:end-1]
            ok &= hi < lo2 || hi2 < lo
        end
    end
    ok && workers_serial(ctx.events)
end

# Run `ntrials` random graphs: serially, checking the order and that each
# graph ran a multiple of its iterations per activation, then on NW workers
# and on more workers than threads, checking they ran the same chunks with
# the same predicate calls and iterations, in a valid order.
function check_random_runs(rng, ntrials::Int, NW::Int)
    for trial in 1:ntrials
        (g, ctx, info) = x_rand_graph(rng, 3)
        x_reset!(ctx, 1)
        ParallelPlans.run!(ParallelPlans.plan(g, 1))
        @test x_ordered(info, ctx)
        @test all(iszero(ctx.iter[id] % info.nrepeat[id]) for id in eachindex(info.nrepeat))
        chunks = x_chunks(ctx)
        iter = copy(ctx.iter)
        calls = copy(ctx.calls)
        for nw in (NW, 2 * Threads.nthreads() + 1)
            x_reset!(ctx, nw)
            ParallelPlans.run!(ParallelPlans.plan(g, nw))
            @test x_chunks(ctx) == chunks
            @test ctx.iter == iter
            @test ctx.calls == calls
            @test x_ordered(info, ctx)
        end
    end
end

@testset "Job graph" begin
    Job = ParallelPlans.Job
    JobGraph = ParallelPlans.JobGraph
    plan = ParallelPlans.plan
    run! = ParallelPlans.run!
    NW = 4

    @testset "construction" begin
        s = Job(job_fun_a, 10)
        @test s.fun === job_fun_a
        @test s.nchunks == 10
        @test Job(job_fun_a, 2^30 - 1).nchunks == 2^30 - 1
        @test Job(job_fun_a, 0).nchunks == 0
        @test_throws ArgumentError Job(job_fun_a, -1)
        @test_throws ArgumentError Job(job_fun_a, 2^30)

        inner = JobGraph([:x => (;job=Job(job_fun_b, 1))])
        g = JobGraph([
            :a => (;job=s),
            :b => (;do_if=pred_a, job=Job(job_fun_b, 2), deps=()),
            :c => (;job=inner, deps=[:b, :a]),
            :d => (;skip_chunks=skip_a, job=inner),
        ]; repeat_if=pred_b)
        @test g.labels == [:a, :b, :c, :d]
        @test g.jobs[1] === s
        @test g.jobs[3] === g.jobs[4] === inner # a graph can be reused
        @test g.deps == [[], [], [2, 1], []] # indices of earlier nodes, decreasing
        # a do_if is normalized to a wrapped skip_chunks
        @test g.skip_chunks == [nothing, ParallelPlans.DoIfSkip(pred_a), nothing, skip_a]
        @test g.repeat_if === pred_b
        @test isnothing(inner.repeat_if)

        # deps implied by another dep, and duplicates, are dropped
        S = Job(job_fun_a, 1)
        @test JobGraph([
            :a => (;job=S),
            :b => (;job=S, deps=[:a]),
            :c => (;job=S, deps=[:a, :b, :a]),
            :d => (;job=S, deps=[:a]),
            :e => (;job=S, deps=[:d, :c]), # both sides of the diamond are kept
        ]).deps == [[], [1], [2], [1], [4, 3]]
        # a lone Symbol is one dep
        @test JobGraph([:a => (;job=S), :b => (;job=S, deps=:a)]).deps == [[], [1]]
    end

    @testset "empty nodes are pruned" begin
        S = Job(job_fun_a, 1)
        S0 = Job(job_fun_b, 0)
        G0 = JobGraph([])
        @test isempty(G0.labels)
        @test isnothing(JobGraph([]; repeat_if=pred_a).repeat_if) # never run, so dropped
        # a graph of only empty nodes is empty, so pruned in turn, and its
        # repeat_if is dropped the same way
        emptied = JobGraph([:a => (;job=S0), :b => (;job=G0, deps=[:a])]; repeat_if=pred_a)
        @test isempty(emptied.labels)
        @test isnothing(emptied.repeat_if)
        @test repr(emptied) == repr(G0) # so an emptied graph prints as the empty one
        # a dep on a pruned node becomes deps on what it depped on, transitively
        g = JobGraph([
            :a => (;job=S),
            :b => (;job=S),
            :c => (;do_if=pred_a, job=S0, deps=[:a]),
            :d => (;job=G0, deps=[:c, :b]),
            :e => (;job=S, deps=[:d]),
            :f => (;job=S, deps=[:c]),
            :g => (;job=S, deps=[:d, :f]), # :a through :d is implied by :f
            :h => (;job=S0),
            :i => (;job=S, deps=[:h]),
        ])
        @test g.labels == [:a, :b, :e, :f, :g, :i]
        @test g.deps == [[], [], [2, 1], [1], [4, 2], []]
        @test g.skip_chunks == fill(nothing, 6)
        # a pruned label is still taken
        @test_throws ArgumentError JobGraph(Any[:a => (;job=S0), :a => (;job=S)])
        # pruned nodes aren't printed, and the deps through them are
        @test !occursin(":c", repr(g))
        @test occursin("deps=[:b, :a]", repr(g))
    end

    @testset "argument errors" begin
        # Any[...] as the constructor compiles once per element type of the
        # node vector, and each of these has its own
        S = Job(job_fun_a, 1)
        @test_throws ArgumentError JobGraph([]; repeat_if=false) # checked though never run
        @test_throws ArgumentError JobGraph(Any[S]) # not a pair
        @test_throws ArgumentError JobGraph(Any["a" => (;job=S)]) # label not a Symbol
        @test_throws ArgumentError JobGraph(Any[:a => S]) # not a NamedTuple
        @test_throws ArgumentError JobGraph(Any[:a => (;nchunks=1)]) # no job
        @test_throws ArgumentError JobGraph(Any[:a => (;job=S, chunks=1)]) # unknown property
        @test_throws ArgumentError JobGraph(Any[:a => (;job=job_fun_a)]) # not a Job
        @test_throws ArgumentError JobGraph(Any[:a => (;do_if=true, job=S)]) # a Bool is not a predicate
        @test_throws ArgumentError JobGraph(Any[:a => (;skip_chunks=1, job=S)]) # a number is not a skip_chunks
        @test_throws ArgumentError JobGraph(Any[:a => (;do_if=pred_a, skip_chunks=skip_a, job=S)]) # not both
        @test_throws ArgumentError JobGraph(Any[:a => (;job=S)]; repeat_if=false)
        @test_throws ArgumentError JobGraph(Any[:a => (;job=S), :a => (;job=S)]) # duplicate label
        @test_throws ArgumentError JobGraph(Any[:a => (;job=S), :b => (;job=S, deps=[1])]) # dep not a Symbol
        @test_throws ArgumentError JobGraph(Any[:a => (;job=S, deps=[:a])]) # dep on itself
        @test_throws ArgumentError JobGraph(Any[:a => (;job=Job(job_fun_a, 0), deps=[:a])]) # even if pruned
        @test_throws ArgumentError JobGraph(Any[:a => (;job=S, deps=[:b]), :b => (;job=S)]) # dep on a later node
        @test_throws ArgumentError plan(JobGraph(Any[:a => (;job=S)]), 0)
    end

    @testset "printing" begin
        g = JobGraph([
            :a => (;job=Job(job_fun_a, 10)),
            :b => (;do_if=pred_a, job=Job(job_fun_b, 1), deps=[:a]),
            :c => (;job=JobGraph([:x => (;job=Job(job_fun_a, 2))]; repeat_if=pred_b), deps=[:a, :b]),
            :d => (;skip_chunks=skip_a, job=Job(job_fun_a, 3), deps=[:c]),
        ])
        s = repr(g)
        @test occursin(r"Job\((Main\.)?job_fun_a, 10\)", s)
        @test occursin(r"do_if=(Main\.)?pred_a", s) # printed as declared, not as its normalization
        @test occursin(r"skip_chunks=(Main\.)?skip_a", s)
        @test occursin(r"repeat_if=(Main\.)?pred_b", s)
        @test occursin("deps=[:b]", s) # :a was dropped, implied by :b
        # callable structs print by type name, as they may hold large state
        ctx = RunCtx()
        s = repr(JobGraph([:a => (;do_if=RunPred(ctx, :a, isodd), job=Job(RunFun(ctx, :a), 3))]; repeat_if=RunPred(ctx, :r, isodd)))
        @test s == "$JobGraph([:a => (;do_if=RunPred, job=$Job(RunFun, 3))]; repeat_if=RunPred)"
        # so do closures, which would print what they capture
        big = zeros(1000)
        s = repr(JobGraph([:a => (;do_if=info -> isempty(big), job=Job((chunk_idx, info) -> sum(big), 1))]))
        @test length(s) < 200 && !occursin("0.0", s)
        @test count("JobGraph([", repr(MIME("text/plain"), g)) == 2
        @test occursin(r"BuiltJobGraph\(@0x[0-9a-f]+, 2 workers\)", repr(plan(g, 2)))
    end

    @testset "error printing" begin
        # with no backtrace, as printing one is slow to compile
        captured = CapturedException(ErrorException("job fun failed"), Any[])
        e = ParallelPlans.JobGraphError([
            ParallelPlans.JobGraphWorkerError(1, [:c, :y], 2, captured),
            ParallelPlans.JobGraphWorkerError(2, [:a], 0, captured), # skip_chunks, a do_if included
            ParallelPlans.JobGraphWorkerError(3, Symbol[], -1, captured), # repeat_if of the root
        ])
        msg = sprint(showerror, e)
        @test occursin("worker 1 in job c.y chunk 2: job fun failed", msg)
        @test occursin("worker 2 in the skip_chunks of a", msg)
        @test occursin("worker 3 in the repeat_if of the root graph", msg)
    end

    @testset "plan" begin
        S1 = Job(job_fun_a, 1)
        inner = JobGraph([
            :x => (;job=Job(job_fun_a, 2)),
            :y => (;do_if=pred_a, job=Job(job_fun_b, 1), deps=[:x]),
            :z => (;job=Job(job_fun_a, 3)),
        ]; repeat_if=pred_b)
        g = JobGraph([
            :a => (;job=Job(job_fun_a, 10)),
            :b => (;job=Job(job_fun_b, 1), deps=[:a]),
            :c => (;do_if=pred_b, job=inner, deps=[:a, :b]),
            :d => (;job=S1, deps=[:c]),
            :e => (;job=inner),
        ])
        b = plan(g, 4)
        @test b.nworkers == 4
        @test length(b.workers) == 4
        root = b.root
        # the children are the declared nodes in order, jobs as is and
        # nested graphs built, one per use
        @test root.labels == [:a, :b, :c, :d, :e]
        @test root.children[1] === g.jobs[1]
        @test root.children[3] isa ParallelPlans.BuiltGraphNode
        @test root.children[3] !== root.children[5]
        @test root.children[3].labels == [:x, :y, :z]
        @test root.skip_chunks == [nothing, nothing, ParallelPlans.DoIfSkip(pred_b), nothing, nothing]
        @test root.children[3].repeat_if === pred_b
        # rev_deps are the later siblings that dep on a child, counts are in
        # chunks with a nested graph counting as one, and the end children
        # are those nothing deps on
        @test root.rev_deps == [[2], [3], [4], [], []]
        @test root.total_dep_chunks == [0, 10, 1, 1, 0]
        @test root.total_end_chunks == 2
        @test length(root.states) == ParallelPlans.JOB_STATE_STRIDE * 7 # a spare stride at each end
        c = root.children[3]
        @test c.rev_deps == [[2], [], []]
        @test c.total_dep_chunks == [0, 2, 0]
        @test c.total_end_chunks == 4
        @test c.states !== root.children[5].states
        # single chunk jobs with deps are on_fin, run by whoever finishes
        # their last dep chunk, and not scanned
        @test root.on_fin == [false, true, false, true, false]
        @test root.scan == [1, 3, 5]
        @test c.on_fin == [false, true, false]
        @test c.scan == [1, 3]
        # but a child gets at most one on_fin rev dep
        b2 = plan(JobGraph([
            :a => (;job=Job(job_fun_a, 2)),
            :b => (;job=S1, deps=[:a]),
            :c => (;job=S1, deps=[:a]),
            :d => (;job=S1, deps=[:b, :c]),
        ]), 2)
        @test b2.root.on_fin == [false, true, false, true]
        @test b2.root.scan == [1, 3]
        # one worker keeps only what the serial run needs
        check_serial_plan(plan(g, 1).root, root)
        # an empty graph plans to a root with no children, and no workers
        b0 = plan(JobGraph([]), 4)
        @test b0.nworkers == 0
        @test isempty(b0.workers)
        @test isempty(b0.root.children)
        check_serial_plan(b0.root, plan(JobGraph([]), 1).root)
    end

    @testset "running pruned nodes" begin
        ctx = RunCtx()
        # an empty graph runs nothing
        @test run_chunks(ctx, JobGraph([]), 1) == []
        @test run_chunks(ctx, JobGraph([]), NW) == []
        # a dep through pruned nodes still orders the run, and the pruned
        # nodes' predicates are never called
        g = JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 4)),
            :b => (;do_if=RunPred(ctx, :b, Returns(true)), job=Job(RunFun(ctx, :b), 0), deps=[:a]),
            :c => (;do_if=RunPred(ctx, :c, Returns(true)), job=JobGraph([
                :x => (;job=Job(RunFun(ctx, :x), 0)),
            ]; repeat_if=RunPred(ctx, :r, Returns(true))), deps=[:b]),
            :d => (;job=Job(RunFun(ctx, :d), 2), deps=[:c]),
            :e => (;job=Job(RunFun(ctx, :e), 3)),
        ])
        chunks = [(:a, 1), (:a, 2), (:a, 3), (:a, 4), (:d, 1), (:d, 2), (:e, 1), (:e, 2), (:e, 3)]
        @test run_chunks(ctx, g, 1) == chunks
        @test isempty(ctx.calls)
        @test run_chunks(ctx, g, NW) == chunks
        @test isempty(ctx.calls)
        @test run_before(ctx, [:a], [:d])
    end

    @testset "chunks, deps and order" begin
        ctx = RunCtx()
        g = JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 4)),
            :b => (;job=Job(RunFun(ctx, :b), 1), deps=[:a]), # on_fin
            :c => (;job=Job(RunFun(ctx, :c), 2), deps=[:b]),
            :d => (;job=Job(RunFun(ctx, :d), 3)),
        ])
        chunks = [(:a, 1), (:a, 2), (:a, 3), (:a, 4), (:b, 1), (:c, 1), (:c, 2), (:d, 1), (:d, 2), (:d, 3)]
        @test run_chunks(ctx, g, 1) == chunks
        @test run_order(ctx) == [:a, :a, :a, :a, :b, :c, :c, :d, :d, :d] # declaration order
        @test run_chunks(ctx, g, NW) == chunks
        @test run_before(ctx, [:a], [:b])
        @test run_before(ctx, [:b], [:c])

        # a plan can be run again, with and without compiling up front
        b = plan(g, NW; compile=false)
        @test run_chunks(ctx, b) == chunks
        @test run_chunks(ctx, b) == chunks
    end

    @testset "do_if" begin
        ctx = RunCtx()
        inner = JobGraph([
            :x => (;job=Job(RunFun(ctx, :x), 2)),
            :y => (;do_if=RunPred(ctx, :y, Returns(true)), job=Job(RunFun(ctx, :y), 1), deps=[:x]),
        ])
        # a skipped job or graph runs nothing, and its dependents proceed
        g = JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 3)),
            :b => (;do_if=RunPred(ctx, :b, Returns(false)), job=Job(RunFun(ctx, :b), 2), deps=[:a]),
            :c => (;do_if=RunPred(ctx, :c, Returns(true)), job=Job(RunFun(ctx, :c), 2), deps=[:a]),
            :d => (;job=Job(RunFun(ctx, :d), 1), deps=[:b, :c]),
            :e => (;do_if=RunPred(ctx, :e, Returns(false)), job=inner, deps=[:d]),
            :f => (;job=Job(RunFun(ctx, :f), 1), deps=[:e]),
        ])
        chunks = [(:a, 1), (:a, 2), (:a, 3), (:c, 1), (:c, 2), (:d, 1), (:f, 1)]
        @test run_chunks(ctx, g, 1) == chunks
        @test ctx.calls == Dict(:b => 1, :c => 1, :e => 1)
        @test run_chunks(ctx, g, NW) == chunks
        @test ctx.calls == Dict(:b => 1, :c => 1, :e => 1)
        @test run_before(ctx, [:a], [:c])
        @test run_before(ctx, [:c], [:d])
        @test run_before(ctx, [:d], [:f])
    end

    @testset "skip_chunks" begin
        ctx = RunCtx()
        inner = JobGraph([
            :x => (;job=Job(RunFun(ctx, :x), 2)),
        ])
        # skip_chunks skips that many chunks from the start, all of them when
        # ≥ nchunks, and the dependents proceed as if the skipped chunks had
        # run. :e2 is an on_fin job skipped, :e and :g on_fin jobs run.
        g = JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 3)),
            :b => (;skip_chunks=RunSkip(ctx, :b, Returns(2)), job=Job(RunFun(ctx, :b), 5), deps=[:a]),
            :c => (;skip_chunks=RunSkip(ctx, :c, Returns(0)), job=Job(RunFun(ctx, :c), 2), deps=[:a]),
            :d => (;skip_chunks=RunSkip(ctx, :d, Returns(typemax(Int))), job=Job(RunFun(ctx, :d), 2), deps=[:a]),
            :e2 => (;skip_chunks=RunSkip(ctx, :e2, Returns(1)), job=Job(RunFun(ctx, :e2), 1), deps=[:c]),
            :e => (;skip_chunks=RunSkip(ctx, :e, Returns(0)), job=Job(RunFun(ctx, :e), 1), deps=[:b, :c, :d, :e2]),
            :f => (;skip_chunks=RunSkip(ctx, :f, Returns(1)), job=inner, deps=[:e]), # a graph is one chunk
            :g => (;skip_chunks=RunSkip(ctx, :g, Returns(0)), job=Job(RunFun(ctx, :g), 1), deps=[:f]),
        ])
        @test ParallelPlans.plan(g, NW).root.on_fin == [false, false, false, false, true, true, false, true]
        chunks = [(:a, 1), (:a, 2), (:a, 3), (:b, 3), (:b, 4), (:b, 5), (:c, 1), (:c, 2), (:e, 1), (:g, 1)]
        calls = Dict(:b => 1, :c => 1, :d => 1, :e2 => 1, :e => 1, :f => 1, :g => 1)
        @test run_chunks(ctx, g, 1) == chunks
        @test ctx.calls == calls
        @test run_chunks(ctx, g, NW) == chunks
        @test ctx.calls == calls
        @test run_before(ctx, [:a], [:b, :c])
        @test run_before(ctx, [:b, :c], [:e])
        @test run_before(ctx, [:e], [:g])
    end

    @testset "repeat_if" begin
        ctx = RunCtx()
        # a graph repeats while its repeat_if is true, checked after each
        # iteration, so it runs at least once; iterations don't overlap
        inner = JobGraph([
            :x => (;job=Job(RunFun(ctx, :x), 2)),
            :y => (;job=Job(RunFun(ctx, :y), 1), deps=[:x]),
        ]; repeat_if=RunPred(ctx, :inner, n -> n < 3))
        g = JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 2)),
            :b => (;job=inner, deps=[:a]),
            :c => (;job=Job(RunFun(ctx, :c), 1), deps=[:b]),
        ])
        chunks = [(:a, 1), (:a, 2), (:c, 1), (:x, 1), (:x, 1), (:x, 1), (:x, 2), (:x, 2), (:x, 2), (:y, 1), (:y, 1), (:y, 1)]
        order = [:a, :a, :x, :x, :y, :x, :x, :y, :x, :x, :y, :c]
        @test run_chunks(ctx, g, 1) == chunks
        @test run_order(ctx) == order
        @test ctx.calls == Dict(:inner => 3)
        @test run_chunks(ctx, g, NW) == chunks
        @test run_order(ctx) == order
        @test ctx.calls == Dict(:inner => 3)

        # the root can repeat, repeats multiply, and do_ifs are reevaluated
        # each iteration
        inner = JobGraph([
            :x => (;job=Job(RunFun(ctx, :x), 1)),
        ]; repeat_if=RunPred(ctx, :inner, isodd))
        g = JobGraph([
            :a => (;do_if=RunPred(ctx, :a, isodd), job=Job(RunFun(ctx, :a), 1)),
            :b => (;job=inner, deps=[:a]),
        ]; repeat_if=RunPred(ctx, :root, n -> n < 3))
        chunks = [(:a, 1), (:a, 1), (:x, 1), (:x, 1), (:x, 1), (:x, 1), (:x, 1), (:x, 1)]
        order = [:a, :x, :x, :x, :x, :a, :x, :x]
        @test run_chunks(ctx, g, 1) == chunks
        @test run_order(ctx) == order
        @test ctx.calls == Dict(:a => 3, :inner => 6, :root => 3)
        @test run_chunks(ctx, g, NW) == chunks
        @test run_order(ctx) == order
        @test ctx.calls == Dict(:a => 3, :inner => 6, :root => 3)

        # a long repeat neither deepens the stack nor loses iterations
        counter = Threads.Atomic{Int}(0)
        bump(chunk_idx, info) = Threads.atomic_add!(counter, 1)
        steps = Ref(0)
        g = JobGraph([
            :a => (;job=Job(bump, 1)),
            :b => (;job=Job(bump, 1), deps=[:a]),
        ]; repeat_if=info -> (steps[] += 1) < 20_000)
        run_long(nw) = (counter[] = 0; steps[] = 0; run!(plan(g, nw)); (steps[], counter[]))
        @test run_long(1) == (20_000, 40_000)
        @test run_long(NW) == (20_000, 40_000)
    end

    @testset "sleeping workers" begin
        ctx = RunCtx()
        # While :a's one chunk runs the other workers run out of spins and
        # sleep. Finishing :a publishes :b's chunks, more than the finisher
        # can take, so it wakes them and they share :b.
        g = JobGraph([
            :a => (;job=Job(SlowFun(RunFun(ctx, :a), 0.05), 1)),
            :b => (;job=Job(SlowFun(RunFun(ctx, :b), 0.01), 4 * NW), deps=[:a]),
        ])
        b = plan(g, NW)
        chunks = [(:a, 1); [(:b, i) for i in 1:4 * NW]]
        @test run_chunks(ctx, b) == chunks
        @test length(unique(e.worker for e in ctx.events if e.id == :b)) > 1
        # every sleeper signed out, and the generation was bumped once, for :b
        @test iszero((@atomic b.notify_sleeper) % UInt16)
        @test (@atomic b.notify_sleeper) ÷ ParallelPlans.ONE_NOTIFY_GEN == 1
        # and again, from the reset counters
        @test run_chunks(ctx, b) == chunks
        @test length(unique(e.worker for e in ctx.events if e.id == :b)) > 1
    end

    @testset "errors" begin
        throw_fun(chunk_idx, info) = error("boom")
        ctx = RunCtx()
        inner = JobGraph([:x => (;job=Job(throw_fun, 3))])
        g = JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 4)),
            :b => (;job=inner, deps=[:a]),
            :c => (;job=Job(RunFun(ctx, :c), 2), deps=[:b]),
        ])
        # serially, an ErrorException naming the job and chunk
        e = run_error(ctx, g, 1)
        @test e isa ErrorException
        @test occursin("in job b.x chunk 1", e.msg)
        @test sort!([(e.id, e.chunk) for e in ctx.events]) == [(:a, 1), (:a, 2), (:a, 3), (:a, 4)]
        # in parallel, a JobGraphError with each erroring worker's error,
        # and no chunk starts after the first error
        e = run_error(ctx, g, NW)
        @test e isa ParallelPlans.JobGraphError
        @test 1 ≤ length(e.worker_errors) ≤ 3
        @test allunique([we.worker_idx for we in e.worker_errors])
        @test all(we.err.ex == ErrorException("boom") for we in e.worker_errors)
        @test all(occursin(r"^in job b.x chunk [123]$", w) for w in error_wheres(e))
        @test sort!([(e.id, e.chunk) for e in ctx.events]) == [(:a, 1), (:a, 2), (:a, 3), (:a, 4)]

        # predicate errors are attributed to the predicate; a do_if is
        # reported as the skip_chunks it is normalized to
        check_pred_error(ctx, JobGraph([
            :a => (;do_if=RunPred(ctx, :a, n -> error("pred")), job=Job(RunFun(ctx, :a), 4)),
        ]), "in the skip_chunks of a", NW)
        # a negative skip_chunks return is an error, attributed to it
        check_pred_error(ctx, JobGraph([
            :a => (;skip_chunks=RunSkip(ctx, :a, Returns(-1)), job=Job(RunFun(ctx, :a), 4)),
        ]), "in the skip_chunks of a", NW)
        check_pred_error(ctx, JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 4)),
        ]; repeat_if=RunPred(ctx, :r, n -> error("rep"))), "in the repeat_if of the root graph", NW)
        check_pred_error(ctx, JobGraph([
            :a => (;job=Job(RunFun(ctx, :a), 4)),
            :b => (;job=JobGraph([:x => (;job=Job(RunFun(ctx, :x), 1))]; repeat_if=RunPred(ctx, :r, n -> error("rep"))), deps=[:a]),
        ]), "in the repeat_if of b", NW)

        # a reused graph is attributed to the node it ran at
        g = JobGraph([
            :c => (;do_if=RunPred(ctx, :c, Returns(false)), job=inner),
            :e => (;job=inner, deps=[:c]),
        ])
        @test occursin("in job e.x chunk 1", run_error(ctx, g, 1).msg)
        @test all(occursin("in job e.x chunk", w) for w in error_wheres(run_error(ctx, g, NW)))

        # a plan that errored runs again
        flag = Ref(true)
        g = JobGraph([
            :a => (;job=Job((chunk_idx, info) -> (flag[] && error("once"); nothing), 2)),
            :b => (;job=Job(RunFun(ctx, :b), 2), deps=[:a]),
        ])
        b = plan(g, NW)
        @test run_error(ctx, b) isa ParallelPlans.JobGraphError
        flag[] = false
        @test run_chunks(ctx, b) == [(:b, 1), (:b, 2)]
    end

    @testset "random nested graphs" begin
        check_random_runs(Random.Xoshiro(0x2b1c), 100, NW)
    end
end
nothing
