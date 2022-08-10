#![doc = include_str!("../Readme.md")]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::suboptimal_flops)]

use once_cell::sync::Lazy;
use primal::{estimate_nth_prime, Sieve};
use rayon::{current_num_threads, prelude::*};
use rug::{
    float::Constant,
    ops::{NegAssign, Pow},
    Float, Rational,
};
use std::{
    cmp::max,
    f64::consts::{LN_2, TAU},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

static SIEVE: Lazy<Sieve> = Lazy::new(|| {
    let (_lo, hi) = estimate_nth_prime(10_000_000);
    Sieve::new(hi as usize)
});

static TABLE_TIME: AtomicU64 = AtomicU64::new(0);
static SUM_TIME: AtomicU64 = AtomicU64::new(0);
static ZETA_TIME: AtomicU64 = AtomicU64::new(0);
static INNER_TIME: AtomicU64 = AtomicU64::new(0);
static DIVSUM_TIME: AtomicU64 = AtomicU64::new(0);

fn zeta(precision: u32, n: u32) -> Float {
    Float::with_val(precision, Float::zeta_u(n))
}

fn zeta_sum(precision: u32, n: u32) -> Float {
    let e = -(n as i32);
    let mut z = Float::with_val(precision, 1);

    // 2^(-n)
    z += Float::with_val(2, Float::i_exp(1, e));

    // Compute upper bound for range
    let k_upper = 2.0.pow((precision as f64) / (n as f64)).ceil() as i32;

    // Sum the terms in parallel with map/reduce
    let sum = (3..=k_upper)
        .into_par_iter()
        .map(|k| {
            let leading_zeros = ((k as f64).log2() * (n as f64) - 1.0).floor() as u32;
            let precision = precision.saturating_sub(leading_zeros).max(32);
            let term = Float::with_val(precision, Float::i_pow_u(k, n)).recip();
            assert!(leading_zeros as i32 <= -term.get_exp().unwrap());
            term
        })
        .reduce(
            || Float::with_val(2, 0),
            |acc, x| {
                if acc.prec() >= x.prec() {
                    acc + &x
                } else {
                    x + &acc
                }
            },
        );
    z += sum;

    z
}

struct ZetaTable {
    precision: u32,
    n:         u32,
    table:     Vec<Float>,
}

impl ZetaTable {
    pub fn new(precision: u32, n: u32) -> Self {
        let mut result = Self {
            precision,
            n,
            table: Vec::new(),
        };
        let k_max = result.k_max();
        result.table.reserve_exact(k_max);
        for k in 3..k_max as u32 {
            result.table.push(result.compute_term(k));
        }
        result
    }

    pub fn zeta_m1(&mut self, n: u32) -> Float {
        assert!(n >= self.n);
        assert!(n % 2 == 0);
        let now = Instant::now();

        // 2^(-n)
        let mut z = Float::new(2);

        while self.n < n {
            self.n += 2;
            let k_upper = self.k_max();

            // 2^(-n)
            z = Float::with_val(self.precision + 1 - n, Float::i_exp(1, -(n as i32)));

            // Update table
            self.table.truncate(k_upper);
            self.table
                .iter_mut()
                .enumerate()
                .skip(3)
                .for_each(|(k, term)| {
                    let leading_zeros = ((k as f64).log2() * (self.n as f64) - 1.0).floor() as u32;
                    let precision = self.precision.saturating_sub(leading_zeros).max(32);
                    term.set_prec(precision);
                    *term /= k * k;
                    z += &*term;
                });
        }
        TABLE_TIME.fetch_add(now.elapsed().as_nanos() as u64, Ordering::Relaxed);
 
        // Test
        // let expected: Float = zeta_sum(self.precision, n) - 1;
        // let error = (&z - expected).abs().to_f64().log2();
        // dbg!(error);

        z
    }

    fn k_max(&self) -> usize {
        2.0.pow((self.precision as f64) / (self.n as f64)).ceil() as usize
    }

    fn compute_term(&self, k: u32) -> Float {
        let leading_zeros = ((k as f64).log2() * (self.n as f64) - 1.0).floor() as u32;
        let precision = self.precision.saturating_sub(leading_zeros).max(32);
        let term = Float::with_val(precision, Float::u_pow_u(k, self.n)).recip();
        assert!(leading_zeros as i32 <= -term.get_exp().unwrap());
        term
    }
}

struct ZetaBoost {
    precision: u32,
    n:         u32,
    tau2:      Float,
    factor:    Float,
    table:     Option<ZetaTable>,
}

impl ZetaBoost {
    pub fn new(precision: u32) -> Self {
        let tau: Float = 2 * Float::with_val(precision, Constant::Pi);
        Self {
            precision,
            n: 0,
            tau2: tau.square(),
            factor: Float::with_val(precision, -0.5),
            table: None,
        }
    }

    pub fn zeta_m1(&mut self, n: u32) -> Float {
        if let Some(table) = &mut self.table {
            let now = Instant::now();
            let result = table.zeta_m1(n);
            ZETA_TIME.fetch_add(now.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return result;
        }

        // Compute required initial precision
        let inner_precision = {
            let n = n as f64;
            40.0 + 2.0 + (n + 1.0) * ((n + 1.0).log2() - 1.0 / LN_2) - n * TAU.log2()
        }
        .max(2.0)
        .ceil() as u32;

        // If there are no savings, start a table approach
        if n >= 3500 || inner_precision >= self.precision {
            self.table = Some(ZetaTable::new(self.precision, n));
            dbg!(n);
            return self.zeta_m1(n);
        }

        // Compute the initial approximation
        let now = Instant::now();
        let zeta = if inner_precision < 10 {
            Float::with_val(inner_precision, 1)
        } else {
            zeta_sum(inner_precision, n)
        };
        INNER_TIME.fetch_add(now.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Compute the factor
        self.update_factor(n);

        // Convert to bernoulli number
        let bernoulli = zeta / &self.factor;

        // Compute the denominator for the Bernoulli number
        let now = Instant::now();
        let divsum = {
            let mut sum = Float::with_val(self.precision, 0);
            // List of primes <https://oeis.org/A080092>
            SIEVE
                .primes_from(2)
                .map(|p| p as u32)
                .take_while(|&p| p - 1 <= n)
                .filter(|&p| n % (p - 1) == 0)
                .for_each(|p| sum += Float::with_val(self.precision, Rational::from((1, p))));
            sum
        };
        DIVSUM_TIME.fetch_add(now.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Turn into integer using von Staudt-Clausen theorem
        // List of integers <https://oeis.org/A000146>
        let integer = {
            let approx = Float::with_val(inner_precision, &bernoulli + &divsum);
            let exact = approx.clone().round();
            let error: f64 = (approx - &exact).abs().to_f64().log2();
            assert!(error < -30.0);
            exact
        };

        // Convert back to zeta using high precision (which we already have for factor
        // and divsum)
        (Float::with_val(self.precision, integer) - divsum) * &self.factor - 1
    }

    // Updatable version of [`factor`]
    fn update_factor(&mut self, n: u32) {
        assert!(n % 2 == 0);
        assert!(n >= self.n);
        while self.n < n {
            self.factor *= &self.tau2;
            self.factor /= (self.n + 1) * (self.n + 2);
            self.factor.neg_assign();
            self.n += 2;
        }
    }

    // Compute the (2π)^n/(2n!) that converts ζ(n) to a Bernoulli number.
    // See <https://en.wikipedia.org/wiki/Particular_values_of_the_Riemann_zeta_function#Even_positive_integers>
    fn factor(&self, n: u32) -> Float {
        let tau: Float = 2 * Float::with_val(self.precision, Constant::Pi);
        let nf = Float::with_val(self.precision, Float::factorial(n));
        let factor: Float = tau.pow(n) / (2 * nf);
        if n % 4 == 0 {
            -factor
        } else {
            factor
        }
    }
}

fn k0(precision: u32) -> Float {
    // ζ(2) = π²/6
    let mut s1: Float = Float::with_val(precision, Constant::Pi).square() / 6 - 1;

    let mut s2: Float = Float::with_val(precision, 1);
    let mut zeta_boost = ZetaBoost::new(precision);
    for n in 2_u32.. {
        let s1_prev = s1.clone();
        s2 -= Rational::from((1, (2 * n - 2) * (2 * n - 1)));
        s1 += &s2 * zeta_boost.zeta_m1(2 * n) / n;
        if s1 == s1_prev {
            break;
        }
    }
    (s1 / Float::with_val(precision, Constant::Log2)).exp()
}

fn bench_zeta(precision: u32) {
    let mut zeta_boost = ZetaBoost::new(precision);
    for i in (4..precision / 2).step_by(2) {
        let now = Instant::now();
        let _result: Float = zeta_boost.zeta_m1(i);
        let elapsed = now.elapsed();
        // let expected: Float = zeta(precision, i) - 1;
        let error = 0.0; //(result - expected).abs().to_f64().log2();
        println!(
            "{} {} {} {}",
            precision,
            i,
            1000.0 * elapsed.as_secs_f64(),
            error
        );
    }
}

fn main() {
    let now = Instant::now();
    SIEVE.is_prime(2); // Make sure the Sieve is hot.
    bench_zeta(34000);

    eprintln!(
        "Table time: {:?}",
        Duration::from_nanos(TABLE_TIME.load(Ordering::Relaxed))
    );
    eprintln!(
        "Sum time: {:?}",
        Duration::from_nanos(SUM_TIME.load(Ordering::Relaxed))
    );
    eprintln!(
        "Zeta time: {:?}",
        Duration::from_nanos(ZETA_TIME.load(Ordering::Relaxed))
    );
    eprintln!(
        "Inner time: {:?}",
        Duration::from_nanos(INNER_TIME.load(Ordering::Relaxed))
    );
    eprintln!(
        "DivSum time: {:?}",
        Duration::from_nanos(DIVSUM_TIME.load(Ordering::Relaxed))
    );
    eprintln!("Total time: {:?}", now.elapsed());

    // for i in 1.. {
    //     let n = i * 512;
    //     println!("{} {}", n, k0(n));
    // }
}
