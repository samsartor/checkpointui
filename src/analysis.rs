use anyhow::{Error, anyhow};
use crossbeam::channel::{Receiver, Sender};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;
use weakref::{Ref, pin};

use crate::model::{ModuleSource, TensorInfo};

pub struct Analysis {
    pub tensor: TensorInfo,
    pub histogram: OnceLock<Histogram>,
    pub error: OnceLock<Error>,
}

pub struct Histogram {
    pub min: f32,
    pub max: f32,
    pub left: f32,
    pub right: f32,
    pub bins: Vec<usize>,
}

const CHECK_EVERY: usize = 1024 * 1024;

pub fn compute_histogram(mut data: Vec<f32>, out: Ref<OnceLock<Histogram>>) {
    if data.is_empty() {
        let _ = out.get(&pin()).unwrap().set(Histogram {
            min: 0.0,
            max: 1.0,
            left: 0.0,
            right: 1.0,
            bins: vec![0],
        });
        return;
    }
    let mut since_check = 0;
    data.sort_unstable_by(|a, b| {
        if since_check >= CHECK_EVERY {
            assert!(out.is_alive());
            since_check = 0;
        }
        since_check += 1;
        let a = if a.is_finite() { *a } else { 0.0 };
        let b = if a.is_finite() { *b } else { 0.0 };
        a.partial_cmp(&b).unwrap()
    });
    assert!(out.is_alive());
    let min = *data.first().unwrap();
    let max = *data.last().unwrap();
    let mut left = min;
    let mut right = max;
    if data.len() >= 10 {
        left = data[data.len() / 10];
        right = data[data.len() / 10 * 9];
        left -= 0.15 * (right - left) / 0.8;
        right += 0.15 * (right - left) / 0.8;
    }
    let bin_count = (data.len() / 5).clamp(5, 1000);
    let mut bins = vec![0usize; bin_count];
    let bins_end = (bin_count - 1) as f32;
    let scale = bins.len() as f32 / (right - left);
    for x in data {
        if since_check >= CHECK_EVERY {
            assert!(out.is_alive());
            since_check = 0;
        }
        since_check += 1;
        let bin = ((x - left) * scale).clamp(0.0, bins_end);
        if !bin.is_finite() {
            continue;
        }
        let bin = bin as usize;
        bins[bin] += 1;
    }
    let _ = out.get(&pin()).unwrap().set(Histogram {
        min,
        max,
        left,
        right,
        bins,
    });
}

fn do_analysis(source: &mut dyn ModuleSource, request: Ref<Analysis>) -> Result<(), Error> {
    let (tensor, cancel, histogram) = {
        let guard = pin();
        let cancel = request.map_with(|_| &(), &guard);
        let histogram = request.map_with(|req| &req.histogram, &guard);
        let request = request.get(&guard).unwrap();
        (request.tensor.clone(), cancel, histogram)
    };
    let data = source.tensor_f32(tensor, cancel)?;
    compute_histogram(data, histogram);
    Ok(())
}

pub fn run_analysis_loop(mut source: Box<dyn ModuleSource>, requests: Receiver<Ref<Analysis>>) {
    loop {
        let Ok(request) = requests.recv() else { return };
        match catch_unwind(AssertUnwindSafe(|| do_analysis(&mut *source, request))) {
            Ok(Ok(_)) => (),
            Ok(Err(err)) => {
                request.inspect(|r| {
                    let _ = r.error.set(err);
                });
            }
            Err(err) => {
                request.inspect(|r| {
                    if let Some(msg) = err.downcast_ref::<&str>() {
                        let _ = r.error.set(anyhow!("panic in analysis: {msg}"));
                    }
                });
            }
        }
    }
}

pub fn start_analysis_thread(source: Box<dyn ModuleSource + Send>) -> Sender<Ref<Analysis>> {
    let (sender, reciever) = crossbeam::channel::bounded(4);
    std::thread::spawn(move || {
        run_analysis_loop(source, reciever);
    });
    sender
}
