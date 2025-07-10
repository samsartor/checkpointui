use anyhow::{Error, anyhow};
use crossbeam::channel::{Receiver, Sender};
use std::panic;
use std::sync::OnceLock;
use weakref::{Ref, pin};

use crate::model::{ModuleSource, TensorInfo};

pub struct Analysis {
    pub tensor: TensorInfo,
    pub max_bin_count: usize,
    pub histogram: OnceLock<Histogram>,
    pub spectrum: OnceLock<Spectrum>,
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

pub fn compute_histogram(data: &[f32], max_bin_count: usize, out: Ref<OnceLock<Histogram>>) {
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
    let mut data = data.to_vec();
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
        left = data[data.len() / 20];
        right = data[19 * data.len() / 20];
        left -= 0.1 * (right - left) / 0.9;
        right += 0.1 * (right - left) / 0.9;
    }
    let bin_count = (data.len() / 5).clamp(5, max_bin_count);
    let mut bins = vec![0usize; bin_count];
    let bins_end = (bin_count - 1) as f32;
    let mut scale = bins.len() as f32 / (right - left);
    scale = if scale.is_finite() { scale } else { 1.0 };
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

pub struct Spectrum {
    pub left: f32,
    pub right: f32,
    pub rank: usize,
    pub bins: Vec<f32>,
}

fn compute_spectrum(
    info: TensorInfo,
    data: &[f32],
    bin_count: usize,
    out: Ref<OnceLock<Spectrum>>,
) {
    if data.is_empty() {
        let _ = out.get(&pin()).unwrap().set(Spectrum {
            left: 0.0,
            right: 1.0,
            rank: 0,
            bins: vec![0.0],
        });
        return;
    }
    let &[h, w] = info.shape.as_slice() else {
        return;
    };
    let h = h as usize;
    let w = w as usize;
    let rank = h.min(w);
    let matrix = faer::MatRef::from_row_major_slice(data, h, w);

    // Compute SVD using nalgebra
    let Ok(mut values) = matrix.singular_values() else {
        return;
    };

    // Convert singular values to histogram
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let left = 0.0;
    let mut right = if values.len() >= 20 {
        values[19 * values.len() / 20]
    } else {
        values.last().copied().unwrap_or(1.0)
    };
    right += 0.15 * right / 0.95;

    let scale = bin_count as f32 / (right - left);
    let mut bins = vec![0.0f32; bin_count];
    let bins_end = (bin_count - 1) as f32;

    // Create histogram with uniform weight (1.0) for each singular value
    for value in values {
        let bin = ((value - left) * scale).clamp(0.0, bins_end);
        if !bin.is_finite() {
            continue;
        }
        let bin = bin as usize;
        bins[bin] += 1.0;
    }

    // Normalize bins to match expected density
    let bin_scale = rank as f32 / bins.iter().sum::<f32>();
    for bin in &mut bins {
        *bin *= bin_scale;
    }

    let _ = out.get(&pin()).unwrap().set(Spectrum {
        left,
        right,
        rank,
        bins,
    });
}

fn do_analysis(source: &mut dyn ModuleSource, request: Ref<Analysis>) -> Result<(), Error> {
    let (tensor, max_bin_count, cancel, histogram, spectrum) = {
        let guard = pin();
        let cancel = request.map_with(|_| &(), &guard);
        let histogram = request.map_with(|req| &req.histogram, &guard);
        let spectrum = request.map_with(|req| &req.spectrum, &guard);
        let request = request.get(&guard).unwrap();
        (
            request.tensor.clone(),
            request.max_bin_count,
            cancel,
            histogram,
            spectrum,
        )
    };
    let data = source.tensor_f32(tensor.clone(), cancel)?;
    compute_histogram(&data, max_bin_count, histogram);
    compute_spectrum(tensor, &data, max_bin_count, spectrum);
    Ok(())
}

pub fn run_analysis_loop(mut source: Box<dyn ModuleSource>, requests: Receiver<Ref<Analysis>>) {
    panic::set_hook(Box::new(|_| {}));
    loop {
        let Ok(request) = requests.recv() else { return };
        match panic::catch_unwind(panic::AssertUnwindSafe(|| {
            do_analysis(&mut *source, request)
        })) {
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
