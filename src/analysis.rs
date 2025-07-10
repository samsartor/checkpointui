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

pub struct BarChart {
    pub bins: Vec<usize>,
    pub left: f32,
    pub right: f32,
    pub continues_past_left: bool,
    pub continues_past_right: bool,
}

impl Default for BarChart {
    fn default() -> Self {
        return BarChart {
            bins: vec![0],
            left: 0.0,
            right: 1.0,
            continues_past_left: true,
            continues_past_right: true,
        };
    }
}

const CHECK_EVERY: usize = 1024 * 1024;

#[derive(Default)]
pub struct Histogram {
    pub min: f32,
    pub max: f32,
    pub chart: BarChart,
}

impl Histogram {
    pub fn new(
        data: &[f32],
        max_bin_count: usize,
        force_min_zero: bool,
        cancel: Ref<()>,
    ) -> Histogram {
        if data.is_empty() {
            return Histogram::default();
        }
        let mut since_check = 0;
        let mut data = data.to_vec();
        data.sort_unstable_by(|a, b| {
            if since_check >= CHECK_EVERY {
                assert!(cancel.is_alive());
                since_check = 0;
            }
            since_check += 1;
            let a = if a.is_finite() { *a } else { 0.0 };
            let b = if a.is_finite() { *b } else { 0.0 };
            a.partial_cmp(&b).unwrap()
        });
        assert!(cancel.is_alive());
        let min = *data.first().unwrap();
        let max = *data.last().unwrap();

        // Calculate display range
        let mut left = if force_min_zero { 0.0 } else { min };
        let mut right = max;

        if data.len() >= 20 {
            // Use 5% and 95% quartiles to estimate range
            let q05 = data[data.len() / 20];
            let q95 = data[19 * data.len() / 20];

            if !force_min_zero {
                // Estimate 0% from 5% and 95% quartiles
                left = q05 - 0.1 * (q95 - q05) / 0.9;
            }

            // Estimate 100% from 5% and 95% quartiles, then add 15%
            right = q95 + 0.1 * (q95 - q05) / 0.9;
            right += 0.15 * right / 0.85;
        }

        let bin_count = (data.len() / 5).clamp(5, max_bin_count);
        let mut bins = vec![0usize; bin_count];
        let bins_end = (bin_count - 1) as f32;
        let mut scale = bins.len() as f32 / (right - left);
        scale = if scale.is_finite() { scale } else { 1.0 };

        // Determine continues_past flags based on range estimation
        let continues_past_left = !force_min_zero && data.len() >= 20;
        let continues_past_right = data.len() >= 20;

        for x in data {
            if since_check >= CHECK_EVERY {
                assert!(cancel.is_alive());
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
        Histogram {
            min,
            max,
            chart: BarChart {
                bins,
                left,
                right,
                continues_past_left,
                continues_past_right,
            },
        }
    }
}

pub struct Spectrum {
    pub chart: BarChart,
}

fn compute_histogram(
    _info: TensorInfo,
    data: &[f32],
    bin_count: usize,
    out: Ref<OnceLock<Histogram>>,
) {
    let guard = pin();
    let _ = out.get(&guard).unwrap().set(Histogram::new(
        &data,
        bin_count,
        false,
        out.map_with(|_| &(), &guard),
    ));
}

fn compute_spectrum(
    info: TensorInfo,
    data: &[f32],
    bin_count: usize,
    out: Ref<OnceLock<Spectrum>>,
) {
    if data.is_empty() {
        let _ = out.get(&pin()).unwrap().set(Spectrum {
            chart: BarChart::default(),
        });
        return;
    }
    let &[h, w] = info.shape.as_slice() else {
        return;
    };
    let h = h as usize;
    let w = w as usize;
    let matrix = faer::MatRef::from_row_major_slice(data, h, w);

    // Compute SVD using faer
    let Ok(values) = matrix.singular_values() else {
        return;
    };

    let guard = pin();
    let _ = out.get(&guard).unwrap().set(Spectrum {
        chart: Histogram::new(&values, bin_count, true, out.map_with(|_| &(), &guard)).chart,
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
    compute_histogram(tensor.clone(), &data, max_bin_count, histogram);
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
