// Synthetic stream source and sink for the shared `parallel` test suite. A
// source is a list of batch sizes; its elements count up from 0 across
// batches, and a negative size is a read failure at that point. A sink records
// every batch it receives. The registries are locked in case the pool serves
// calls on several threads.

struct MlcpartestSource { sizes: Vec<i64>, i: usize, start: i64 }

static MLCPARTEST_SOURCES: std::sync::Mutex<Vec<MlcpartestSource>> = std::sync::Mutex::new(Vec::new());
static MLCPARTEST_SINKS: std::sync::Mutex<Vec<Vec<Vec<i64>>>> = std::sync::Mutex::new(Vec::new());

pub fn mlcpartest_synth_open(sizes: &Vec<i64>) -> i64 {
    let mut s = MLCPARTEST_SOURCES.lock().unwrap();
    s.push(MlcpartestSource { sizes: sizes.clone(), i: 0, start: 0 });
    (s.len() - 1) as i64
}

pub fn mlcpartest_synth_next(k: i64) -> (bool, Vec<i64>) {
    let mut all = MLCPARTEST_SOURCES.lock().unwrap();
    let s = &mut all[k as usize];
    if s.i >= s.sizes.len() { return (true, Vec::new()); }
    let n = s.sizes[s.i];
    s.i += 1;
    if n < 0 { return (false, Vec::new()); }
    let xs: Vec<i64> = (s.start..s.start + n).collect();
    s.start += n;
    (true, xs)
}

pub fn mlcpartest_sink_open() -> i64 {
    let mut s = MLCPARTEST_SINKS.lock().unwrap();
    s.push(Vec::new());
    (s.len() - 1) as i64
}

pub fn mlcpartest_sink_put(k: i64, xs: &Vec<i64>) {
    MLCPARTEST_SINKS.lock().unwrap()[k as usize].push(xs.clone());
}

pub fn mlcpartest_sink_batches(k: i64) -> Vec<Vec<i64>> {
    MLCPARTEST_SINKS.lock().unwrap()[k as usize].clone()
}
