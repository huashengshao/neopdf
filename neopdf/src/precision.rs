//! Scalar-generic interpolation of ordinary collinear grids.
//!
//! This module evaluates the same log-Hermite interpolant used by NeoPDF without
//! restricting its arithmetic to `f64`. The input dataset is still binary64:
//! elevated precision stabilizes evaluation of that fixed dataset, not the
//! physical accuracy of its fit. Knots, logarithms, slopes and coefficients are
//! rebuilt in the caller's arithmetic, rather than promoting rounded results.
//!
//! The supported class is a strictly increasing, positive rectangular `(x,Q²)`
//! grid and a strictly increasing tabulated coupling scale. Extrapolation,
//! positivity clipping, and matching across threshold subgrids are caller-owned
//! policies. In particular duplicate threshold knots are rejected, not removed.

use std::cmp::Ordering;
use thiserror::Error;

/// Arithmetic context for interpolation, independent of any multiprecision crate.
///
/// The context controls precision and rounding. `lift` must convert the *exact*
/// binary64 input to that arithmetic; it must not round through a decimal string.
/// Operations must use the context's arithmetic, including constants, and
/// propagate any backend accuracy tracking. Results may carry fewer reliable
/// bits after cancellation; these must not be silently restored by promotion.
pub trait Arithmetic {
    /// A finite real scalar; it need not implement `Copy`.
    type Scalar: Clone;
    /// Convert fixed binary64 input data at the context's precision.
    fn lift(&self, value: f64) -> Self::Scalar;
    /// Add two scalars.
    fn add(&self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    /// Subtract two scalars.
    fn sub(&self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    /// Multiply two scalars.
    fn mul(&self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    /// Divide two scalars.
    fn div(&self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    /// Natural logarithm in the context's arithmetic.
    fn ln(&self, a: &Self::Scalar) -> Self::Scalar;
    /// Compare without conversion to binary64. Unordered values return `None`.
    fn compare(&self, a: &Self::Scalar, b: &Self::Scalar) -> Option<Ordering>;
    /// Whether the scalar is finite.
    fn is_finite(&self, a: &Self::Scalar) -> bool;
}

/// Standard binary64 arithmetic, useful for parity checks and ordinary clients.
#[derive(Clone, Copy, Debug, Default)]
pub struct Binary64;

impl Arithmetic for Binary64 {
    type Scalar = f64;
    fn lift(&self, v: f64) -> f64 {
        v
    }
    fn add(&self, a: &f64, b: &f64) -> f64 {
        a + b
    }
    fn sub(&self, a: &f64, b: &f64) -> f64 {
        a - b
    }
    fn mul(&self, a: &f64, b: &f64) -> f64 {
        a * b
    }
    fn div(&self, a: &f64, b: &f64) -> f64 {
        a / b
    }
    fn ln(&self, a: &f64) -> f64 {
        a.ln()
    }
    fn compare(&self, a: &f64, b: &f64) -> Option<Ordering> {
        a.partial_cmp(b)
    }
    fn is_finite(&self, a: &f64) -> bool {
        a.is_finite()
    }
}

/// Errors are explicit; failed interpolation never substitutes a zero PDF.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum InterpolationError {
    /// Every axis needs at least two knots, and grids need at least one flavour.
    #[error("insufficient grid dimensions")]
    InvalidShape,
    /// Axes must be positive, finite, and strictly increasing.
    #[error("axis knots must be positive, finite and strictly increasing")]
    InvalidAxis,
    /// The requested arithmetic cannot distinguish consecutive transformed knots.
    #[error("working precision cannot resolve the grid knots")]
    InsufficientPrecision,
    /// All stored values and evaluated results must be finite.
    #[error("nonfinite interpolation data, query or result")]
    NonFinite,
    /// This API deliberately has no implicit extrapolation policy.
    #[error("query is outside the supported grid")]
    OutOfRange,
    /// The flavour slot is outside this grid's stored range.
    #[error("flavour slot is not present in the grid")]
    InvalidFlavor,
}

/// Shared Hermite polynomial, also used by NeoPDF's existing binary64 API.
#[inline]
pub fn hermite<A: Arithmetic>(
    a: &A,
    t: &A::Scalar,
    vl: &A::Scalar,
    vdl: &A::Scalar,
    vh: &A::Scalar,
    vdh: &A::Scalar,
) -> A::Scalar {
    let t2 = a.mul(t, t);
    let t3 = a.mul(&t2, t);
    let two_t3 = a.mul(&a.lift(2.0), &t3);
    let three_t2 = a.mul(&a.lift(3.0), &t2);
    let p0 = a.mul(&a.add(&a.sub(&two_t3, &three_t2), &a.lift(1.0)), vl);
    let m0 = a.mul(&a.add(&a.sub(&t3, &a.mul(&a.lift(2.0), &t2)), t), vdl);
    let p1 = a.mul(&a.add(&a.sub(&a.lift(0.0), &two_t3), &three_t2), vh);
    let m1 = a.mul(&a.sub(&t3, &t2), vdh);
    a.add(&a.add(&a.add(&p0, &m0), &p1), &m1)
}

/// The same averaged secant slope used by the collinear and αs kernels.
#[inline]
pub(crate) fn derivative<A: Arithmetic>(
    a: &A,
    knots: &[A::Scalar],
    values: &[A::Scalar],
    i: usize,
) -> A::Scalar {
    let slope = |j: usize| {
        a.div(
            &a.sub(&values[j + 1], &values[j]),
            &a.sub(&knots[j + 1], &knots[j]),
        )
    };
    if i == 0 {
        slope(0)
    } else if i + 1 == knots.len() {
        slope(i - 1)
    } else {
        a.mul(&a.add(&slope(i - 1), &slope(i)), &a.lift(0.5))
    }
}

struct Axis<T> {
    knots: Vec<T>,
    logs: Vec<T>,
}

impl<T: Clone> Axis<T> {
    fn new<A: Arithmetic<Scalar = T>>(
        a: &A,
        raw: &[f64],
        square: bool,
    ) -> Result<Self, InterpolationError> {
        if raw.len() < 2 {
            return Err(InterpolationError::InvalidShape);
        }
        if raw.iter().any(|x| !x.is_finite() || *x <= 0.0) || raw.windows(2).any(|p| p[0] >= p[1]) {
            return Err(InterpolationError::InvalidAxis);
        }
        let knots: Vec<_> = raw
            .iter()
            .map(|x| {
                let v = a.lift(*x);
                if square {
                    a.mul(&v, &v)
                } else {
                    v
                }
            })
            .collect();
        let logs: Vec<_> = knots.iter().map(|x| a.ln(x)).collect();
        if knots.iter().chain(&logs).any(|x| !a.is_finite(x))
            || knots
                .windows(2)
                .chain(logs.windows(2))
                .any(|p| a.compare(&p[0], &p[1]) != Some(Ordering::Less))
        {
            return Err(InterpolationError::InsufficientPrecision);
        }
        Ok(Self { knots, logs })
    }

    fn locate<A: Arithmetic<Scalar = T>>(
        &self,
        a: &A,
        x: &T,
    ) -> Result<(usize, T), InterpolationError> {
        if !a.is_finite(x) {
            return Err(InterpolationError::NonFinite);
        }
        if a.compare(x, &self.knots[0]) == Some(Ordering::Less)
            || a.compare(x, self.knots.last().unwrap()) == Some(Ordering::Greater)
        {
            return Err(InterpolationError::OutOfRange);
        }
        let mut lo = 0;
        let mut hi = self.knots.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match a.compare(&self.knots[mid], x) {
                Some(Ordering::Less | Ordering::Equal) => lo = mid + 1,
                Some(Ordering::Greater) => hi = mid,
                None => return Err(InterpolationError::NonFinite),
            }
        }
        if lo == 0 {
            return Err(InterpolationError::OutOfRange);
        }
        let i = (lo - 1).min(self.knots.len() - 2);
        let t = a.div(
            &a.sub(&a.ln(x), &self.logs[i]),
            &a.sub(&self.logs[i + 1], &self.logs[i]),
        );
        if !a.is_finite(&t) {
            return Err(InterpolationError::NonFinite);
        }
        Ok((i, t))
    }
}

/// A one-dimensional log-Hermite table, including tabulated αs.
pub struct LogCubicTable<A: Arithmetic> {
    arithmetic: A,
    axis: Axis<A::Scalar>,
    values: Vec<A::Scalar>,
    derivatives: Vec<A::Scalar>,
}

impl<A: Arithmetic> LogCubicTable<A> {
    /// Build from positive physical coordinates and values.
    pub fn new(a: A, coordinates: &[f64], values: &[f64]) -> Result<Self, InterpolationError> {
        Self::build(a, coordinates, values, false)
    }

    /// Build αs(Q²) from metadata's unsquared Q knots.
    ///
    /// Squaring is performed in the selected arithmetic, not in binary64.
    pub fn from_q_knots(a: A, qs: &[f64], values: &[f64]) -> Result<Self, InterpolationError> {
        Self::build(a, qs, values, true)
    }

    fn build(
        a: A,
        coordinates: &[f64],
        values: &[f64],
        square: bool,
    ) -> Result<Self, InterpolationError> {
        if coordinates.len() != values.len() {
            return Err(InterpolationError::InvalidShape);
        }
        let axis = Axis::new(&a, coordinates, square)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(InterpolationError::NonFinite);
        }
        let values: Vec<_> = values.iter().map(|v| a.lift(*v)).collect();
        let derivatives = (0..values.len())
            .map(|i| derivative(&a, &axis.logs, &values, i))
            .collect();
        Ok(Self {
            arithmetic: a,
            axis,
            values,
            derivatives,
        })
    }

    /// Evaluate without extrapolation or a conversion to binary64.
    pub fn evaluate(&self, coordinate: &A::Scalar) -> Result<A::Scalar, InterpolationError> {
        self.evaluate_with_interval(coordinate)
            .map(|(value, _)| value)
    }

    /// Evaluate and return the actual lower-endpoint index of the interval used.
    ///
    /// Both results come from the same lookup and interpolation operation. An
    /// exact interior knot selects the interval to its right; the final knot
    /// selects the last interval. Both outer endpoints are included, with no
    /// tolerance snapping or conversion of the query to binary64. For a table
    /// built from Q knots, the index refers to their squares in the selected
    /// arithmetic. Metadata is returned only when evaluation succeeds.
    pub fn evaluate_with_interval(
        &self,
        coordinate: &A::Scalar,
    ) -> Result<(A::Scalar, usize), InterpolationError> {
        let a = &self.arithmetic;
        let (i, t) = self.axis.locate(a, coordinate)?;
        let dx = a.sub(&self.axis.logs[i + 1], &self.axis.logs[i]);
        let result = hermite(
            a,
            &t,
            &self.values[i],
            &a.mul(&self.derivatives[i], &dx),
            &self.values[i + 1],
            &a.mul(&self.derivatives[i + 1], &dx),
        );
        if a.is_finite(&result) {
            Ok((result, i))
        } else {
            Err(InterpolationError::NonFinite)
        }
    }
}

/// The actual polynomial cell used by a successful collinear grid evaluation.
///
/// Indices identify lower endpoints in the immutable input grid, not knot
/// values or an independently recomputed query location. Their interpretation
/// therefore requires the identity and preparation options of that grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridCell {
    /// Lower-endpoint index along the x axis.
    pub x_interval: usize,
    /// Lower-endpoint index along the Q-squared axis.
    pub q2_interval: usize,
}

/// A scalar-generic rectangular collinear log-Hermite grid.
///
/// Values are `x*f(x,Q²)`; the caller chooses flavour slots and any positivity
/// policy. This type does not silently extrapolate or identify missing flavours
/// with zero. It uses NeoPDF's averaged-secant derivatives in both dimensions.
pub struct LogBicubicGrid<A: Arithmetic> {
    arithmetic: A,
    x: Axis<A::Scalar>,
    q2: Axis<A::Scalar>,
    flavors: usize,
    // [x-cell][flavor][Q²-row][polynomial coefficient, descending power]
    coefficients: Vec<[A::Scalar; 4]>,
}

/// Representation of fixed scale knots before preparation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScaleAxis {
    /// Already-squared fixed Q² data.
    #[default]
    QSquared,
    /// Original Q data, squared in the selected arithmetic.
    Q,
}

/// Interpolation rule when a Q segment has exactly two knots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TwoPointQ {
    /// Keep cubic x interpolation and use a linear Q dependence.
    #[default]
    CubicX,
    /// Use linear interpolation in both logarithmic coordinates, as LHAPDF's
    /// log-cubic interpolator does for two-Q-knot threshold segments.
    LogBilinear,
}

/// Explicit input-coordinate and short-segment interpolation policies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridOptions {
    /// Whether the input scale data are Q or Q².
    pub scale_axis: ScaleAxis,
    /// Rule for a Q segment containing only two knots.
    pub two_point_q: TwoPointQ,
}

impl<A: Arithmetic> LogBicubicGrid<A> {
    /// Rebuild from fixed raw grid values in the requested arithmetic.
    ///
    /// `value_at(flavor, x_index, q2_index)` returns a raw dataset entry, not
    /// a previously interpolated value or cached binary64 polynomial coefficient.
    pub fn new<F: Fn(usize, usize, usize) -> f64>(
        a: A,
        xs: &[f64],
        q2s: &[f64],
        flavors: usize,
        value_at: F,
    ) -> Result<Self, InterpolationError> {
        Self::with_options(a, xs, q2s, flavors, value_at, GridOptions::default())
    }

    /// Rebuild from unsquared Q knots retained from the original input file.
    ///
    /// This avoids rounding Q² in binary64 before selecting the arithmetic.
    /// In particular PDF and αs thresholds specified by the same Q then remain
    /// identical at arbitrary precision.
    pub fn from_q_knots<F: Fn(usize, usize, usize) -> f64>(
        a: A,
        xs: &[f64],
        qs: &[f64],
        flavors: usize,
        value_at: F,
    ) -> Result<Self, InterpolationError> {
        Self::with_options(
            a,
            xs,
            qs,
            flavors,
            value_at,
            GridOptions {
                scale_axis: ScaleAxis::Q,
                ..GridOptions::default()
            },
        )
    }

    /// Build with explicit coordinate and two-knot policies.
    pub fn with_options<F: Fn(usize, usize, usize) -> f64>(
        a: A,
        xs: &[f64],
        q2s: &[f64],
        flavors: usize,
        value_at: F,
        options: GridOptions,
    ) -> Result<Self, InterpolationError> {
        if flavors == 0 {
            return Err(InterpolationError::InvalidShape);
        }
        let x = Axis::new(&a, xs, false)?;
        let q2 = Axis::new(&a, q2s, options.scale_axis == ScaleAxis::Q)?;
        let count = (xs.len() - 1)
            .checked_mul(flavors)
            .and_then(|n| n.checked_mul(q2s.len()))
            .ok_or(InterpolationError::InvalidShape)?;
        let mut coefficients = vec![[a.lift(0.0), a.lift(0.0), a.lift(0.0), a.lift(0.0)]; count];
        for flavor in 0..flavors {
            for iq in 0..q2s.len() {
                let raw: Vec<_> = (0..xs.len()).map(|ix| value_at(flavor, ix, iq)).collect();
                if raw.iter().any(|v| !v.is_finite()) {
                    return Err(InterpolationError::NonFinite);
                }
                let values: Vec<_> = raw.into_iter().map(|v| a.lift(v)).collect();
                for ix in 0..xs.len() - 1 {
                    if q2s.len() == 2 && options.two_point_q == TwoPointQ::LogBilinear {
                        coefficients[(ix * flavors + flavor) * q2s.len() + iq] = [
                            a.lift(0.0),
                            a.lift(0.0),
                            a.sub(&values[ix + 1], &values[ix]),
                            values[ix].clone(),
                        ];
                        continue;
                    }
                    let dx = a.sub(&x.logs[ix + 1], &x.logs[ix]);
                    let dl = a.mul(&derivative(&a, &x.logs, &values, ix), &dx);
                    let dh = a.mul(&derivative(&a, &x.logs, &values, ix + 1), &dx);
                    let vl = &values[ix];
                    let vh = &values[ix + 1];
                    let ca = a.add(
                        &a.sub(&a.add(&dh, &dl), &a.mul(&a.lift(2.0), vh)),
                        &a.mul(&a.lift(2.0), vl),
                    );
                    let cb = a.sub(
                        &a.sub(
                            &a.sub(&a.mul(&a.lift(3.0), vh), &a.mul(&a.lift(3.0), vl)),
                            &a.mul(&a.lift(2.0), &dl),
                        ),
                        &dh,
                    );
                    coefficients[(ix * flavors + flavor) * q2s.len() + iq] =
                        [ca, cb, dl, vl.clone()];
                }
            }
        }
        Ok(Self {
            arithmetic: a,
            x,
            q2,
            flavors,
            coefficients,
        })
    }

    /// Evaluate `x*f` at an in-grid coordinate, retaining the scalar type.
    pub fn evaluate(
        &self,
        flavor: usize,
        x: &A::Scalar,
        q2: &A::Scalar,
    ) -> Result<A::Scalar, InterpolationError> {
        self.evaluate_with_cell(flavor, x, q2)
            .map(|(value, _)| value)
    }

    /// Evaluate `x*f` and return the actual cell used by this operation.
    ///
    /// Lookup and interpolation occur once in the selected arithmetic. On each
    /// axis, an exact interior knot selects the interval to its right, while
    /// the final knot selects the last interval. Both outer endpoints are
    /// included, without tolerance snapping or narrowing the query. If the
    /// grid was built from Q knots, the Q-squared index refers to their squares
    /// in that arithmetic. Caller-owned segment, flavor-identity and positivity
    /// policies are not inferred here. Metadata is returned only on success.
    pub fn evaluate_with_cell(
        &self,
        flavor: usize,
        x: &A::Scalar,
        q2: &A::Scalar,
    ) -> Result<(A::Scalar, GridCell), InterpolationError> {
        if flavor >= self.flavors {
            return Err(InterpolationError::InvalidFlavor);
        }
        let a = &self.arithmetic;
        let (ix, u) = self.x.locate(a, x)?;
        let (iq, v) = self.q2.locate(a, q2)?;
        let nq = self.q2.logs.len();
        let base = (ix * self.flavors + flavor) * nq;
        let eval_x = |j: usize| {
            let c = &self.coefficients[base + j];
            // The same x-polynomial as InterleavedHermite::hermite_x.
            let u2 = a.mul(&u, &u);
            let u3 = a.mul(&u2, &u);
            a.add(
                &a.add(
                    &a.add(&a.mul(&c[0], &u3), &a.mul(&c[1], &u2)),
                    &a.mul(&c[2], &u),
                ),
                &c[3],
            )
        };
        let lo = iq.saturating_sub(1);
        let hi = (iq + 2).min(nq - 1);
        let values: Vec<_> = (lo..=hi).map(eval_x).collect();
        let knots = &self.q2.logs[lo..=hi];
        let j = iq - lo;
        let dq = a.sub(&self.q2.logs[iq + 1], &self.q2.logs[iq]);
        let dl = a.mul(&derivative(a, knots, &values, j), &dq);
        let dh = a.mul(&derivative(a, knots, &values, j + 1), &dq);
        let result = hermite(a, &v, &values[j], &dl, &values[j + 1], &dh);
        if a.is_finite(&result) {
            Ok((
                result,
                GridCell {
                    x_interval: ix,
                    q2_interval: iq,
                },
            ))
        } else {
            Err(InterpolationError::NonFinite)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleaved::InterleavedHermite;

    #[test]
    fn collinear_grid_matches_existing_algorithm_at_edges_and_interiors() {
        let xs: Vec<f64> = vec![0.001, 0.007, 0.13, 0.51, 1.0];
        let qs: Vec<f64> = vec![2.0, 13.0, 81.0, 500.0];
        let value =
            |f: usize, i: usize, j: usize| (f + 1) as f64 * (xs[i].sqrt() + qs[j].ln().sin());
        let generic = LogBicubicGrid::new(Binary64, &xs, &qs, 3, value).unwrap();
        let legacy = InterleavedHermite::build(
            xs.iter().map(|x| x.ln()).collect(),
            vec![qs.iter().map(|q| q.ln()).collect()],
            3,
            |f, i, j| value(f, i, j[0]),
        );
        for f in 0..3 {
            for x in [0.001, 0.0021, 0.007, 0.08, 0.13, 0.4, 0.51, 0.83, 1.0] {
                for q in [2.0, 4.0, 13.0, 30.0, 81.0, 300.0, 500.0] {
                    let got = generic.evaluate(f, &x, &q).unwrap();
                    let expected = legacy.eval_single(f, &[x, q]);
                    assert!(
                        (got - expected).abs() < 2e-13,
                        "{f} {x} {q}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn log_affine_functions_are_reproduced_and_two_knots_are_supported() {
        let xs: Vec<f64> = vec![0.01, 1.0];
        let qs: Vec<f64> = vec![1.0, 100.0];
        let grid = LogBicubicGrid::new(Binary64, &xs, &qs, 1, |_, i, j| {
            2.0 + 3.0 * xs[i].ln() - qs[j].ln()
        })
        .unwrap();
        for (x, q) in [(0.01, 1.0), (0.03, 7.0), (0.4, 70.0), (1.0, 100.0)] {
            assert!(
                (grid.evaluate(0, &x, &q).unwrap() - (2.0 + 3.0 * x.ln() - q.ln())).abs() < 1e-13
            );
        }
        let table = LogCubicTable::from_q_knots(Binary64, &[1.0, 10.0], &[0.2, 0.1]).unwrap();
        assert!((table.evaluate(&10.0).unwrap() - 0.15).abs() < 1e-15);
    }

    #[test]
    fn invalid_inputs_and_unsupported_thresholds_fail_explicitly() {
        assert!(matches!(
            LogCubicTable::new(Binary64, &[1.0, 1.0], &[0.1, 0.2]),
            Err(InterpolationError::InvalidAxis)
        ));
        assert!(matches!(
            LogCubicTable::new(Binary64, &[0.0, 1.0], &[0.1, 0.2]),
            Err(InterpolationError::InvalidAxis)
        ));
        assert!(matches!(
            LogCubicTable::new(Binary64, &[1.0], &[0.1]),
            Err(InterpolationError::InvalidShape)
        ));
        assert!(matches!(
            LogCubicTable::new(Binary64, &[1.0, 2.0], &[0.1, f64::NAN]),
            Err(InterpolationError::NonFinite)
        ));
        let grid =
            LogBicubicGrid::new(Binary64, &[0.1, 1.0], &[1.0, 10.0], 1, |_, _, _| 0.0).unwrap();
        assert_eq!(
            grid.evaluate(0, &0.01, &2.0),
            Err(InterpolationError::OutOfRange)
        );
        assert_eq!(
            grid.evaluate(0, &0.5, &11.0),
            Err(InterpolationError::OutOfRange)
        );
        assert_eq!(
            grid.evaluate(0, &f64::NAN, &2.0),
            Err(InterpolationError::NonFinite)
        );
        assert_eq!(
            grid.evaluate(1, &0.5, &2.0),
            Err(InterpolationError::InvalidFlavor)
        );
        assert_eq!(grid.evaluate(0, &0.5, &2.0), Ok(0.0));
    }

    #[test]
    fn explicit_two_q_knot_policy_uses_log_bilinear_fallback() {
        let xs = [0.125, 0.25, 0.5, 1.0];
        let qs = [1.4, 4.0];
        let grid = LogBicubicGrid::with_options(
            Binary64,
            &xs,
            &qs,
            1,
            |_, ix, iq| (ix * ix + iq) as f64,
            GridOptions {
                scale_axis: ScaleAxis::Q,
                two_point_q: TwoPointQ::LogBilinear,
            },
        )
        .unwrap();
        // Both query coordinates lie midway in log space between two knots.
        let x = (xs[1] * xs[2]).sqrt();
        let q2 = qs[0] * qs[1];
        assert!((grid.evaluate(0, &x, &q2).unwrap() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn located_values_follow_log_affine_oracles_and_endpoint_ownership() {
        let knots: [f64; 4] = [1.0, 2.0, 4.0, 8.0];
        let table = LogCubicTable::new(Binary64, &knots, &[7.0, 9.0, 11.0, 13.0]).unwrap();
        let queries = [
            (1.0, 0),
            (1.5, 0),
            (f64::from_bits(2.0_f64.to_bits() - 1), 0),
            (2.0, 1),
            (f64::from_bits(2.0_f64.to_bits() + 1), 1),
            (3.0, 1),
            (4.0, 2),
            (6.0, 2),
            (8.0, 2),
        ];
        for (query, expected_interval) in queries {
            let (value, interval) = table.evaluate_with_interval(&query).unwrap();
            assert_eq!(interval, expected_interval);
            assert!((value - (7.0 + 2.0 * query.log2())).abs() < 1e-13);
            assert_eq!(value, table.evaluate(&query).unwrap());
        }

        let xs: [f64; 4] = [0.125, 0.25, 0.5, 1.0];
        let grid = LogBicubicGrid::new(Binary64, &xs, &knots, 2, |flavor, ix, iq| {
            3.0 + flavor as f64 + 2.0 * xs[ix].log2() - knots[iq].log2()
        })
        .unwrap();
        for (x, x_interval) in [
            (0.125, 0),
            (0.2, 0),
            (f64::from_bits(0.25_f64.to_bits() - 1), 0),
            (0.25, 1),
            (f64::from_bits(0.25_f64.to_bits() + 1), 1),
            (0.375, 1),
            (0.5, 2),
            (0.75, 2),
            (1.0, 2),
        ] {
            for (q2, q2_interval) in queries {
                for flavor in 0..2 {
                    let (value, cell) = grid.evaluate_with_cell(flavor, &x, &q2).unwrap();
                    assert_eq!(
                        cell,
                        GridCell {
                            x_interval,
                            q2_interval
                        }
                    );
                    let expected = 3.0 + flavor as f64 + 2.0 * x.log2() - q2.log2();
                    assert!((value - expected).abs() < 1e-13);
                    assert_eq!(value, grid.evaluate(flavor, &x, &q2).unwrap());
                }
            }
        }
    }

    #[test]
    fn equal_values_do_not_hide_distinct_lookup_cells() {
        let table = LogCubicTable::new(Binary64, &[1.0, 2.0, 4.0], &[0.0; 3]).unwrap();
        assert_eq!(table.evaluate_with_interval(&1.5).unwrap(), (0.0, 0));
        assert_eq!(table.evaluate_with_interval(&3.0).unwrap(), (0.0, 1));
        let grid = LogBicubicGrid::new(
            Binary64,
            &[0.125, 0.25, 1.0],
            &[1.0, 2.0, 4.0],
            1,
            |_, _, _| 0.0,
        )
        .unwrap();
        for (x, x_interval) in [(0.2, 0), (0.5, 1)] {
            for (q2, q2_interval) in [(1.5, 0), (3.0, 1)] {
                assert_eq!(
                    grid.evaluate_with_cell(0, &x, &q2).unwrap(),
                    (
                        0.0,
                        GridCell {
                            x_interval,
                            q2_interval
                        }
                    )
                );
            }
        }
    }

    #[test]
    fn located_values_retain_original_q_and_two_knot_policies() {
        let qs: [f64; 3] = [1.4, 4.0, 8.0];
        let table = LogCubicTable::from_q_knots(Binary64, &qs, &[0.3, 0.2, 0.1]).unwrap();
        for (q2, expected_interval) in [
            (qs[0] * qs[0], 0),
            (f64::from_bits((qs[1] * qs[1]).to_bits() - 1), 0),
            (qs[1] * qs[1], 1),
            (qs[2] * qs[2], 1),
        ] {
            let (value, interval) = table.evaluate_with_interval(&q2).unwrap();
            assert_eq!(interval, expected_interval);
            assert_eq!(value, table.evaluate(&q2).unwrap());
        }
        let xs: [f64; 4] = [0.125, 0.25, 0.5, 1.0];
        let grid = LogBicubicGrid::with_options(
            Binary64,
            &xs,
            &qs[..2],
            1,
            |_, ix, iq| (ix * ix + iq) as f64,
            GridOptions {
                scale_axis: ScaleAxis::Q,
                two_point_q: TwoPointQ::LogBilinear,
            },
        )
        .unwrap();
        let x = (xs[1] * xs[2]).sqrt();
        let q2 = qs[0] * qs[1];
        let (value, cell) = grid.evaluate_with_cell(0, &x, &q2).unwrap();
        assert_eq!(
            cell,
            GridCell {
                x_interval: 1,
                q2_interval: 0
            }
        );
        assert!((value - 3.0).abs() < 1e-14);
        for (x, q2, expected) in [
            (
                xs[0],
                qs[0] * qs[0],
                GridCell {
                    x_interval: 0,
                    q2_interval: 0,
                },
            ),
            (
                xs[3],
                qs[1] * qs[1],
                GridCell {
                    x_interval: 2,
                    q2_interval: 0,
                },
            ),
        ] {
            let (value, cell) = grid.evaluate_with_cell(0, &x, &q2).unwrap();
            assert_eq!(cell, expected);
            assert_eq!(value, grid.evaluate(0, &x, &q2).unwrap());
        }
    }

    #[test]
    fn metadata_evaluation_retains_error_categories_and_order() {
        let table = LogCubicTable::new(Binary64, &[1.0, 2.0], &[0.3, 0.2]).unwrap();
        for (query, error) in [
            (0.5, InterpolationError::OutOfRange),
            (3.0, InterpolationError::OutOfRange),
            (f64::NAN, InterpolationError::NonFinite),
            (f64::INFINITY, InterpolationError::NonFinite),
        ] {
            assert_eq!(table.evaluate_with_interval(&query).unwrap_err(), error);
            assert_eq!(table.evaluate(&query).unwrap_err(), error);
        }
        let grid =
            LogBicubicGrid::new(Binary64, &[0.125, 1.0], &[1.0, 4.0], 1, |_, _, _| 0.0).unwrap();
        for (flavor, x, q2, error) in [
            (1, f64::NAN, f64::NAN, InterpolationError::InvalidFlavor),
            (0, 0.01, f64::NAN, InterpolationError::OutOfRange),
            (0, f64::NAN, 5.0, InterpolationError::NonFinite),
            (0, 0.5, 5.0, InterpolationError::OutOfRange),
            (0, 0.5, f64::NAN, InterpolationError::NonFinite),
        ] {
            assert_eq!(grid.evaluate_with_cell(flavor, &x, &q2).unwrap_err(), error);
            assert_eq!(grid.evaluate(flavor, &x, &q2).unwrap_err(), error);
        }
    }

    #[test]
    fn metadata_is_collected_without_repeating_query_logarithms() {
        use std::{cell::Cell, rc::Rc};
        #[derive(Clone)]
        struct CountLogs(Rc<Cell<usize>>);
        impl Arithmetic for CountLogs {
            type Scalar = f64;
            fn lift(&self, value: f64) -> f64 {
                value
            }
            fn add(&self, a: &f64, b: &f64) -> f64 {
                a + b
            }
            fn sub(&self, a: &f64, b: &f64) -> f64 {
                a - b
            }
            fn mul(&self, a: &f64, b: &f64) -> f64 {
                a * b
            }
            fn div(&self, a: &f64, b: &f64) -> f64 {
                a / b
            }
            fn ln(&self, value: &f64) -> f64 {
                self.0.set(self.0.get() + 1);
                value.ln()
            }
            fn compare(&self, a: &f64, b: &f64) -> Option<Ordering> {
                a.partial_cmp(b)
            }
            fn is_finite(&self, value: &f64) -> bool {
                value.is_finite()
            }
        }
        let calls = Rc::new(Cell::new(0));
        let table =
            LogCubicTable::new(CountLogs(calls.clone()), &[1.0, 2.0, 4.0], &[0.3, 0.2, 0.1])
                .unwrap();
        calls.set(0);
        let (located, interval) = table.evaluate_with_interval(&1.5).unwrap();
        assert_eq!(interval, 0);
        assert_eq!(calls.replace(0), 1);
        assert_eq!(table.evaluate(&1.5).unwrap(), located);
        assert_eq!(calls.replace(0), 1);
        let grid = LogBicubicGrid::new(
            CountLogs(calls.clone()),
            &[0.125, 0.25, 1.0],
            &[1.0, 2.0, 4.0],
            1,
            |_, ix, iq| (ix + iq) as f64,
        )
        .unwrap();
        calls.set(0);
        let (located, cell) = grid.evaluate_with_cell(0, &0.5, &1.5).unwrap();
        assert_eq!(
            cell,
            GridCell {
                x_interval: 1,
                q2_interval: 0
            }
        );
        assert_eq!(calls.replace(0), 2);
        assert_eq!(grid.evaluate(0, &0.5, &1.5).unwrap(), located);
        assert_eq!(calls.get(), 2);
    }
}
