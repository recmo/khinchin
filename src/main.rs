#![doc = include_str!("../Readme.md")]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::suboptimal_flops)]

use once_cell::sync::Lazy;
use primal::{estimate_nth_prime, Sieve};
use rug::{float::Constant, ops::Pow, Float};
use std::{
    f64::consts::{LN_2, TAU},
    time::Instant,
};

static SIEVE: Lazy<Sieve> = Lazy::new(|| {
    let (_lo, hi) = estimate_nth_prime(10_000_000);
    Sieve::new(hi as usize)
});

fn zeta(precision: u32, n: u32) -> Float {
    Float::with_val(precision, Float::zeta_u(n))
}

fn zeta_sum(precision: u32, n: u32) -> Float {
    let e = -(n as i32);
    let mut z = Float::with_val(precision, 1);

    // 2^(-n)
    z += Float::with_val(2, Float::i_exp(1, e));

    (3..)
        .map(|k| {
            let leading_zeros = (((k as f64).log2() * (n as f64) - 1.0).floor() as u32);
            (k, leading_zeros)
        })
        .take_while(|(_, leading_zeros)| leading_zeros <= &precision)
        .map(|(k, leading_zeros)| {
            let precision = (precision - leading_zeros).max(32);
            let term = Float::with_val(precision, k).pow(e);
            assert!(leading_zeros as i32 <= -term.get_exp().unwrap());
            term
        })
        .for_each(|term| z += term);

    z
}

/// Computes ζ(n) using a lower precision version.
///
/// von Staudt-Clausen's algorithm.
fn zeta_boost<F>(inner: F, precision: u32, n: u32) -> Float
where
    F: FnOnce(u32, u32) -> Float,
{
    // Compute required initial precision
    let inner_precision = {
        let n = n as f64;
        40.0 + 2.0 + (n + 1.0) * ((n + 1.0).log2() - 1.0 / LN_2) - n * TAU.log2()
    }
    .max(2.0)
    .ceil() as u32;

    // If there are no savings, skip
    if inner_precision >= precision {
        return inner(precision, n);
    }

    // Compute the initial approximation
    let zeta = if inner_precision < 10 {
        Float::with_val(inner_precision, 1)
    } else {
        inner(inner_precision, n)
    };

    // Compute the (2π)^n/(2n!) that converts ζ(n) to a Bernoulli number.
    // See <https://en.wikipedia.org/wiki/Particular_values_of_the_Riemann_zeta_function#Even_positive_integers>
    let factor = {
        let tau: Float = 2 * Float::with_val(precision, Constant::Pi);
        let nf = Float::with_val(precision, Float::factorial(n));
        let factor: Float = tau.pow(n) / (2 * nf);
        if n % 4 == 0 {
            -factor
        } else {
            factor
        }
    };

    // Convert to bernoulli number
    let bernoulli = zeta / &factor;

    // Compute the denominator for the Bernoulli number
    let divsum = {
        let mut sum = Float::with_val(precision, 0);
        // List of primes <https://oeis.org/A080092>
        SIEVE
            .primes_from(2)
            .map(|p| p as u32)
            .take_while(|&p| p - 1 <= n)
            .filter(|&p| n % (p - 1) == 0)
            .for_each(|p| sum += Float::with_val(precision, p).recip());
        sum
    };

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
    (Float::with_val(precision, integer) - divsum) * factor
}

fn k0(precision: u32) -> Float {
    // ζ(2) = π²/6
    let mut s1: Float = Float::with_val(precision, Constant::Pi).square() / 6 - 1;

    let mut s2: Float = Float::with_val(precision, 1);
    for n in 2_u32.. {
        let s1_prev = s1.clone();
        s2 -= Float::with_val(precision, (2 * n - 2) * (2 * n - 1)).recip();
        s1 += &s2 * (zeta_boost(zeta_sum, precision, 2 * n) - 1) / n;
        if s1 == s1_prev {
            break;
        }
    }
    (s1 / Float::with_val(precision, Constant::Log2)).exp()
}

fn bench_zeta(precision: u32) {
    for i in (4..precision / 2).step_by(2) {
        let now = Instant::now();
        let result: Float = zeta_boost(zeta_sum, precision, i) - 1;
        let elapsed = now.elapsed();
        // let expected: Float = zeta(precision, i) - 1;
        // let error = (result - expected).abs().to_f64().log2();
        println!("{} {} {}", precision, i, 1000.0 * elapsed.as_secs_f64(),);
    }
}

fn main() {
    SIEVE.is_prime(2); // Make sure the Sieve is hot.
    bench_zeta(34000);
    // for i in 1.. {
    //     let n = i * 512;
    //     println!("{} {}", n, k0(n));
    // }
}
