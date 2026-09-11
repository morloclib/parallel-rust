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
fn mlcpar_run_spans<B, Body>(opts: &ParOpts, spans: &[(usize, usize)], body: Body) -> Vec<Vec<B>>
where
    B: Send,
    Body: Fn(usize) -> Vec<B> + Sync,
{
    if spans.is_empty() { return Vec::new(); }
    if spans.len() == 1 { return vec![body(0)]; }

    let nthreads = mlcpar_workers(opts).min(spans.len());
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let gathered = std::sync::Mutex::new(Vec::<(usize, Vec<B>)>::with_capacity(spans.len()));

    std::thread::scope(|scope| {
        for _ in 0..nthreads {
            scope.spawn(|| {
                let mut local: Vec<(usize, Vec<B>)> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= spans.len() { break; }
                    local.push((i, body(i)));
                }
                gathered.lock().unwrap().append(&mut local);
            });
        }
    });

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
