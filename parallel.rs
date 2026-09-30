// Rust implementations of the morloc `parallel` module.
//
// Workers are scoped threads pulling from a shared cursor, so a shrinking
// schedule is genuinely dynamic rather than merely uneven. Scoped threads let
// a worker borrow the input slice directly: no clone of the payload, which for
// a list of sequences is the dominant cost.

// User-mapped types, matching the declarations in parallel/main.loc. Field and
// tag order are the wire contract: a record is positional and a constructor's
// tag is its position, so reordering either here silently misreads values.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParChunking { EvenChunks = 0, ShrinkingChunks = 1, FixedChunks = 2 }

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParOrder { InputOrder = 0, ArrivalOrder = 1 }

#[derive(Clone, Debug)]
pub struct ParOpts {
    pub workers: Option<i64>,
    pub chunkSize: Option<i64>,
    pub chunking: ParChunking,
    pub order: ParOrder,
    pub inflight: Option<i64>,
}

fn mlcpar_positive_or(v: &Option<i64>, fallback: usize) -> usize {
    match v {
        Some(n) if *n > 0 => *n as usize,
        _ => fallback,
    }
}

fn mlcpar_workers(opts: &ParOpts) -> usize {
    let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    mlcpar_positive_or(&opts.workers, hw).max(1)
}

/// Split [0, n) into work units. Identical in shape to the Python and C++
/// backends: the schedule is part of the module's meaning, not of any one
/// language's implementation.
fn mlcpar_ranges(n: usize, opts: &ParOpts) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if n == 0 { return out; }
    let w = mlcpar_workers(opts);

    match opts.chunking {
        ParChunking::EvenChunks => {
            let size = n.div_ceil(w);
            let mut i = 0;
            while i < n { out.push((i, (i + size).min(n))); i += size; }
        }
        ParChunking::FixedChunks => {
            let size = mlcpar_positive_or(&opts.chunkSize, 1).max(1);
            let mut i = 0;
            while i < n { out.push((i, (i + size).min(n))); i += size; }
        }
        ParChunking::ShrinkingChunks => {
            // Each round deals w units of half the remaining work, so early
            // units are large and the tail is single elements.
            let mut lo = 0;
            while lo < n {
                let size = ((n - lo) / (2 * w)).max(1);
                for _ in 0..w {
                    if lo >= n { break; }
                    let hi = (lo + size).min(n);
                    out.push((lo, hi));
                    lo = hi;
                }
            }
        }
    }
    out
}

/// Run `body` over every span on threads pulling from a shared cursor, and
/// return the per-span results in span order.
///
/// Results are gathered as (index, value) and sorted rather than written into
/// a preallocated slice, because that needs no unsafe and no per-slot lock:
/// the sort is over one entry per work unit, not per element.
///
/// A panicking unit is caught in its worker and its payload is resumed on the
/// calling thread. Letting it escape instead would make the scope re-panic
/// with a generic payload, losing the error the caller's `@try` looks for.
fn mlcpar_run_spans<B, Body>(opts: &ParOpts, spans: &[(usize, usize)], body: Body) -> Vec<Vec<B>>
where
    B: Send,
    Body: Fn(usize) -> Vec<B> + Sync,
{
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    if spans.is_empty() { return Vec::new(); }
    if spans.len() == 1 { return vec![body(0)]; }

    let nthreads = mlcpar_workers(opts).min(spans.len());
    let cursor = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let failure = std::sync::Mutex::new(None::<Box<dyn std::any::Any + Send>>);
    let gathered = std::sync::Mutex::new(Vec::<(usize, Vec<B>)>::with_capacity(spans.len()));

    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| {
                let mut local: Vec<(usize, Vec<B>)> = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= spans.len() { break; }
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(i))) {
                        Ok(v) => local.push((i, v)),
                        Err(payload) => {
                            stop.store(true, Ordering::Relaxed);
                            failure.lock().unwrap().get_or_insert(payload);
                            break;
                        }
                    }
                }
                gathered.lock().unwrap().append(&mut local);
            });
        }
    });

    if let Some(payload) = failure.into_inner().unwrap() {
        std::panic::resume_unwind(payload);
    }

    let mut out = gathered.into_inner().unwrap();
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, v)| v).collect()
}

fn mlcpar_flatten<B>(parts: Vec<Vec<B>>) -> Vec<B> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts { out.extend(p); }
    out
}

pub fn morloc_pmap_with<A, B, F>(opts: &ParOpts, f: F, xs: &[A]) -> Vec<B>
where
    A: Sync,
    B: Send,
    F: Fn(&A) -> B + Sync,
{
    let spans = mlcpar_ranges(xs.len(), opts);
    let parts = mlcpar_run_spans(opts, &spans, |i| {
        let (lo, hi) = spans[i];
        xs[lo..hi].iter().map(|x| f(x)).collect()
    });
    mlcpar_flatten(parts)
}

pub fn morloc_pconcat_map_with<A, B, F>(opts: &ParOpts, f: F, xs: &[A]) -> Vec<B>
where
    A: Sync,
    B: Send,
    F: Fn(&A) -> Vec<B> + Sync,
{
    let spans = mlcpar_ranges(xs.len(), opts);
    let parts = mlcpar_run_spans(opts, &spans, |i| {
        let (lo, hi) = spans[i];
        let mut out = Vec::new();
        for x in &xs[lo..hi] { out.extend(f(x)); }
        out
    });
    mlcpar_flatten(parts)
}

pub fn morloc_pfilter_with<A, P>(opts: &ParOpts, pred: P, xs: &[A]) -> Vec<A>
where
    A: Sync + Clone,
    P: Fn(&A) -> bool + Sync,
{
    let spans = mlcpar_ranges(xs.len(), opts);
    let parts = mlcpar_run_spans(opts, &spans, |i| {
        let (lo, hi) = spans[i];
        xs[lo..hi].iter().map(|x| pred(x)).collect::<Vec<bool>>()
    });
    // The predicate runs in parallel; the compaction is a sequential pass, so
    // survivors keep their input order without a merge step.
    let keep = mlcpar_flatten(parts);
    xs.iter().zip(keep).filter(|(_, k)| *k).map(|(x, _)| x.clone()).collect()
}


// Streams
// -------
//
// A stream stage is a pipeline. The calling thread pulls batches and sinks
// results -- the source and sink are morloc closures bound to this thread --
// while scoped worker threads map work units. `inflight` counts units
// dispatched and not yet delivered, so it bounds the reorder buffer as well as
// the queue.
//
// A panic on the calling thread (in the pull, the sink or a combine) drops the
// work sender, so the workers see the disconnect and exit, and the scope's
// join completes.

type MlcparUnit<A> = (usize, std::sync::Arc<Vec<A>>, usize, usize);
type MlcparDone<B> = (usize, std::thread::Result<Vec<B>>);

/// Run a stream stage. `compute(slice)` maps one unit; `deliver` receives each
/// non-empty unit result, in input order when `ordered`. Returns (true, "") at
/// end of stream or (false, msg) after a read failure, once every dispatched
/// unit has been delivered. A panic in a unit is resumed on the calling thread
/// with its own payload.
fn mlcpar_pipeline<A, B, P, C, D>(
    opts: &ParOpts,
    pull: &P,
    compute: C,
    mut deliver: D,
    ordered: bool,
) -> (bool, String)
where
    A: Send + Sync,
    B: Send,
    P: Fn() -> (bool, String, Vec<A>),
    C: Fn(&[A]) -> Vec<B> + Sync,
    D: FnMut(Vec<B>),
{
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc, Mutex};

    let w = mlcpar_workers(opts);
    if w <= 1 {
        loop {
            let (ok, msg, xs) = pull();
            if !ok { return (false, msg); }
            if xs.is_empty() { return (true, String::new()); }
            for (lo, hi) in mlcpar_ranges(xs.len(), opts) {
                let ys = compute(&xs[lo..hi]);
                if !ys.is_empty() { deliver(ys); }
            }
        }
    }

    let limit = mlcpar_positive_or(&opts.inflight, 2 * w).max(1);
    let stop = AtomicBool::new(false);

    // The workers borrow the work queue, so it outlives the scope; the senders
    // are moved into the scope, so a panic on the calling thread drops them,
    // the workers see the disconnect and exit, and the scope's join completes.
    let (work_tx, work_rx) = mpsc::channel::<MlcparUnit<A>>();
    let work_rx = Mutex::new(work_rx);
    let (done_tx, done_rx) = mpsc::channel::<MlcparDone<B>>();
    let (status, failure) = std::thread::scope(|scope| {
        let work_tx = work_tx;
        let done_tx = done_tx;
        for _ in 0..w {
            let done_tx = done_tx.clone();
            let (work_rx, compute, stop) = (&work_rx, &compute, &stop);
            scope.spawn(move || loop {
                let unit = work_rx.lock().unwrap().recv();
                let (seq, batch, lo, hi) = match unit { Ok(u) => u, Err(_) => return };
                let result = if stop.load(Ordering::Relaxed) {
                    Ok(Vec::new())
                } else {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute(&batch[lo..hi])))
                };
                if done_tx.send((seq, result)).is_err() { return; }
            });
        }
        drop(done_tx);

        let mut batch: Option<Arc<Vec<A>>> = None;
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut span_i = 0;
        let mut reading = true;
        let mut status = (true, String::new());
        let mut failure: Option<Box<dyn std::any::Any + Send>> = None;
        let (mut next_seq, mut next_out, mut running, mut undelivered) = (0usize, 0usize, 0usize, 0usize);
        let mut held: std::collections::BTreeMap<usize, Vec<B>> = std::collections::BTreeMap::new();

        loop {
            while reading && failure.is_none() && undelivered < limit {
                if span_i == spans.len() {
                    let (ok, msg, xs) = pull();
                    if !ok { reading = false; status = (false, msg); break; }
                    if xs.is_empty() { reading = false; break; }
                    spans = mlcpar_ranges(xs.len(), opts);
                    span_i = 0;
                    batch = Some(Arc::new(xs));
                }
                let (lo, hi) = spans[span_i];
                span_i += 1;
                let b = Arc::clone(batch.as_ref().unwrap());
                if work_tx.send((next_seq, b, lo, hi)).is_err() { break; }
                next_seq += 1;
                running += 1;
                undelivered += 1;
            }

            if running == 0 { break; }

            let (seq, result) = match done_rx.recv() { Ok(d) => d, Err(_) => break };
            running -= 1;
            match result {
                Err(payload) => {
                    stop.store(true, Ordering::Relaxed);
                    if failure.is_none() { failure = Some(payload); }
                }
                Ok(_) if failure.is_some() => {}
                Ok(ys) if ordered => {
                    held.insert(seq, ys);
                    while let Some(ys) = held.remove(&next_out) {
                        next_out += 1;
                        undelivered -= 1;
                        if !ys.is_empty() { deliver(ys); }
                    }
                }
                Ok(ys) => {
                    undelivered -= 1;
                    if !ys.is_empty() { deliver(ys); }
                }
            }
        }
        drop(work_tx);
        (status, failure)
    });

    if let Some(payload) = failure { std::panic::resume_unwind(payload); }
    status
}

pub fn morloc_psconcat_map_native<A, B, F, P, S, R>(opts: &ParOpts, f: F, pull: P, sink: S) -> (bool, String)
where
    A: Send + Sync,
    B: Send,
    F: Fn(&A) -> Vec<B> + Sync,
    P: Fn() -> (bool, String, Vec<A>),
    S: Fn(&Vec<B>) -> R,
{
    let compute = |xs: &[A]| {
        let mut out = Vec::new();
        for x in xs { out.extend(f(x)); }
        out
    };
    mlcpar_pipeline(opts, &pull, compute, |ys| { sink(&ys); }, opts.order == ParOrder::InputOrder)
}

pub fn morloc_psmap_native<A, B, F, P, S, R>(opts: &ParOpts, f: F, pull: P, sink: S) -> (bool, String)
where
    A: Send + Sync,
    B: Send,
    F: Fn(&A) -> B + Sync,
    P: Fn() -> (bool, String, Vec<A>),
    S: Fn(&Vec<B>) -> R,
{
    let compute = |xs: &[A]| xs.iter().map(|x| f(x)).collect::<Vec<B>>();
    mlcpar_pipeline(opts, &pull, compute, |ys| { sink(&ys); }, opts.order == ParOrder::InputOrder)
}

pub fn morloc_psfilter_native<A, F, P, S, R>(opts: &ParOpts, pred: F, pull: P, sink: S) -> (bool, String)
where
    A: Send + Sync + Clone,
    F: Fn(&A) -> bool + Sync,
    P: Fn() -> (bool, String, Vec<A>),
    S: Fn(&Vec<A>) -> R,
{
    let compute = |xs: &[A]| xs.iter().filter(|x| pred(x)).cloned().collect::<Vec<A>>();
    mlcpar_pipeline(opts, &pull, compute, |ys| { sink(&ys); }, opts.order == ParOrder::InputOrder)
}

/// The fold always consumes results in input order, whatever `order` asks: a
/// fixed fold order is what makes the answer schedule-independent. The
/// identity arrives by value or by reference depending on its type, hence
/// `Borrow`.
pub fn morloc_psfold_native<A, B, I, C, F, P>(
    opts: &ParOpts,
    combine: C,
    identity: I,
    f: F,
    pull: P,
) -> (bool, String, B)
where
    A: Send + Sync,
    B: Send + Clone,
    I: std::borrow::Borrow<B>,
    C: Fn(&B, &B) -> B,
    F: Fn(&A) -> B + Sync,
    P: Fn() -> (bool, String, Vec<A>),
{
    let mut acc: B = identity.borrow().clone();
    let compute = |xs: &[A]| xs.iter().map(|x| f(x)).collect::<Vec<B>>();
    let (ok, msg) = mlcpar_pipeline(opts, &pull, compute, |ys| {
        for y in ys.iter() { acc = combine(&acc, y); }
    }, true);
    (ok, msg, acc)
}
