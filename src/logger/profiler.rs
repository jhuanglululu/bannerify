// profiler
pub fn init_profiler() {
    #[cfg(feature = "profiling")]
    crate::logger::profiler::timing_internal::init_clock();
}

// timing

#[cfg(feature = "profiling")]
macro_rules! timed {
    ($label:expr) => {{
        $crate::logger::profile_print(format!(
            "{} in {} ms",
            $label,
            $crate::logger::profiler::timing_internal::since_last().as_millis(),
        ));
    }};
}

#[cfg(not(feature = "profiling"))]
macro_rules! timed {
    ($label:expr) => {};
}

#[cfg(feature = "profiling")]
macro_rules! finish_profiling {
    () => {{
        $crate::logger::profile_print(format!(
            "program finished in {:.3} seconds",
            $crate::logger::profiler::timing_internal::elapsed().as_secs_f64(),
        ));

        $crate::logger::profile_print(format!(
            "peak memory usage: {:.3} mb",
            $crate::logger::profiler::ALLOCATOR.peak_usage_as_mb()
        ));
    }};
}

#[cfg(not(feature = "profiling"))]
macro_rules! finish_profiling {
    () => {};
}

pub(crate) use {finish_profiling, timed};

/// this is guard behind profiling feature
#[cfg(feature = "profiling")]
pub mod timing_internal {
    use std::sync::{LazyLock, RwLock};
    use std::time::{Duration, Instant};

    static ELAPSED: LazyLock<Instant> = LazyLock::new(Instant::now);
    static LAST: RwLock<Option<Instant>> = RwLock::new(None);

    pub fn init_clock() {
        LazyLock::force(&ELAPSED);
        *LAST.write().unwrap() = Some(Instant::now());
    }

    pub fn elapsed() -> Duration {
        ELAPSED.elapsed()
    }

    pub fn since_last() -> Duration {
        let now = Instant::now();
        let prev = LAST.write().unwrap().replace(now).unwrap();
        now.duration_since(prev)
    }
}

// allocators

#[cfg(feature = "profiling")]
#[global_allocator]
pub static ALLOCATOR: crate::allocator::PeakAlloc = crate::allocator::PeakAlloc;

#[cfg(not(feature = "profiling"))]
#[global_allocator]
pub static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;
