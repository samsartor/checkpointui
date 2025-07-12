use anyhow::{Error, anyhow, bail};
use async_cell::sync::{AsyncCell, GetRef, TakeRef};
use futures_lite::future::block_on;
use rand::seq::SliceRandom;
use std::sync::OnceLock;
use weakref::{Ref, pin};

use crate::model::{ModuleSource, TensorInfo};

#[derive(Debug, Clone)]
pub enum Req<T> {
    Requested,
    Completed(T),
}

pub type ReqCell<T> = AsyncCell<Req<T>>;

pub struct Analysis {
    pub tensor: TensorInfo,
    pub max_bin_count: usize,
    pub histogram: ReqCell<Histogram>,
    pub spectrum: ReqCell<Spectrum>,
    pub error: OnceLock<Error>,
}

impl Drop for Analysis {
    fn drop(&mut self) {
        eprintln!("dropped analysis");
    }
}

#[derive(Debug, Clone)]
pub struct BarChart {
    pub bins: Vec<usize>,
    pub left: f32,
    pub right: f32,
    pub continues_past_left: bool,
    pub continues_past_right: bool,
}

impl Default for BarChart {
    fn default() -> Self {
        BarChart {
            bins: vec![0],
            left: 0.0,
            right: 1.0,
            continues_past_left: true,
            continues_past_right: true,
        }
    }
}

const QUARTILE_SAMPLES: usize = 200;

#[derive(Default, Debug, Clone)]
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
    ) -> Result<Histogram, Error> {
        if data.is_empty() {
            bail!("tensor is empty");
        }

        // For large datasets, use random sampling to estimate quantiles
        let sample_data = if data.len() > QUARTILE_SAMPLES {
            let mut rng = rand::thread_rng();
            data.choose_multiple(&mut rng, QUARTILE_SAMPLES)
                .copied()
                .collect()
        } else {
            data.to_vec()
        };

        // Sort the sample (much smaller now)
        let mut sorted_sample = sample_data.clone();
        sorted_sample.sort_unstable_by(|a, b| {
            let a = if a.is_finite() { *a } else { 0.0 };
            let b = if a.is_finite() { *b } else { 0.0 };
            a.partial_cmp(&b).unwrap()
        });
        if !cancel.is_alive() {
            bail!("canceled");
        }

        // Find actual min/max from full dataset
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Calculate display range
        let mut left = if force_min_zero { 0.0 } else { min };
        let mut right = max;

        if sorted_sample.len() >= QUARTILE_SAMPLES {
            // Use 5% and 95% percentiles to estimate range
            let p05_idx = ((sorted_sample.len() - 1) as f32 * 0.05) as usize;
            let p95_idx = ((sorted_sample.len() - 1) as f32 * 0.95) as usize;
            let q05 = sorted_sample[p05_idx];
            let q95 = sorted_sample[p95_idx];

            if !force_min_zero {
                // Estimate 0% from 5% and 95% percentiles
                left = q05 - 0.05 * (q95 - q05) / 0.90;
            }

            // Estimate 100% from 5% and 95% percentiles, then add 15%
            right = q95 + 0.05 * (q95 - q05) / 0.90;
            right += 0.15 * right / 0.85;
        }

        let bin_count = (data.len() / 5).clamp(5, max_bin_count);
        let mut bins = vec![0usize; bin_count];
        let bins_end = (bin_count - 1) as f32;
        let mut scale = bins.len() as f32 / (right - left);
        scale = if scale.is_finite() { scale } else { 1.0 };

        // Determine continues_past flags based on range estimation
        let continues_past_left = !force_min_zero && sorted_sample.len() >= QUARTILE_SAMPLES;
        let continues_past_right = sorted_sample.len() >= QUARTILE_SAMPLES;

        for x in data {
            let bin = ((x - left) * scale).clamp(0.0, bins_end);
            if !bin.is_finite() {
                continue;
            }
            let bin = bin as usize;
            bins[bin] += 1;
        }
        Ok(Histogram {
            min,
            max,
            chart: BarChart {
                bins,
                left,
                right,
                continues_past_left,
                continues_past_right,
            },
        })
    }
}

#[derive(Default, Debug, Clone)]
pub struct Spectrum {
    pub chart: BarChart,
}

fn compute_histogram(
    _info: TensorInfo,
    data: &[f32],
    bin_count: usize,
    out: Ref<ReqCell<Histogram>>,
) -> Result<(), Error> {
    match block_on(GetRef(out)) {
        None => bail!("cancelled"),
        Some(Req::Requested) => (),
        Some(Req::Completed(_)) => bail!("already computed"),
    };
    let histogram = Histogram::new(data, bin_count, false, out.map(|_| &()))?;
    {
        out.get(&pin())
            .ok_or(anyhow!("cancelled"))?
            .set(Req::Completed(histogram));
    }
    Ok(())
}

fn compute_spectrum(
    info: TensorInfo,
    data: &[f32],
    bin_count: usize,
    out: Ref<ReqCell<Spectrum>>,
) -> Result<(), Error> {
    match block_on(GetRef(out)) {
        None => bail!("cancelled"),
        Some(Req::Requested) => (),
        Some(Req::Completed(_)) => bail!("already computed"),
    };

    if data.is_empty() {
        out.get(&pin())
            .ok_or(anyhow!("cancelled"))?
            .set(Req::Completed(Spectrum {
                chart: BarChart::default(),
            }));
        bail!("tensor is empty");
    }

    let &[h, w] = info.shape.as_slice() else {
        return Ok(());
    };
    let h = h as usize;
    let w = w as usize;
    let matrix = faer::MatRef::from_row_major_slice(data, h, w);

    // Compute SVD using faer
    let values = matrix
        .singular_values()
        .map_err(|err| anyhow!("could not perform SVD: {err:?}"))?;
    let histogram = Histogram::new(&values, bin_count, true, out.map(|_| &()))?;
    {
        out.get(&pin())
            .ok_or(anyhow!("cancelled"))?
            .set(Req::Completed(Spectrum {
                chart: histogram.chart,
            }));
    }
    Ok(())
}

fn do_analysis(source: &mut dyn ModuleSource, request: Ref<Analysis>) -> Result<(), Error> {
    let (tensor, max_bin_count, cancel, histogram, spectrum) = {
        let guard = pin();
        let cancel = request.map_with(|_| &(), &guard);
        let histogram = request.map_with(|req| &req.histogram, &guard);
        let spectrum = request.map_with(|req| &req.spectrum, &guard);
        let request = request.get(&guard).ok_or(anyhow!("cancelled"))?;
        (
            request.tensor.clone(),
            request.max_bin_count,
            cancel,
            histogram,
            spectrum,
        )
    };
    let data = source.tensor_f32(tensor.clone(), cancel)?;
    compute_histogram(tensor.clone(), &data, max_bin_count, histogram)?;
    compute_spectrum(tensor, &data, max_bin_count, spectrum)?;
    Ok(())
}

pub type AnalysisCell = AsyncCell<Ref<Analysis>>;

pub fn run_analysis_loop(mut source: Box<dyn ModuleSource>, requests: Ref<AnalysisCell>) {
    loop {
        let Some(request) = block_on(TakeRef(requests)) else {
            return;
        };
        match do_analysis(&mut *source, request) {
            Ok(_) => (),
            Err(err) => {
                request.inspect(|r| {
                    let _ = r.error.set(err);
                });
            }
        }
    }
}

pub fn start_analysis_thread(source: Box<dyn ModuleSource + Send>, cell: Ref<AnalysisCell>) {
    std::thread::spawn(move || {
        run_analysis_loop(source, cell);
    });
}
