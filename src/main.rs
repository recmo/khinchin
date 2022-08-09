#![doc = include_str!("../Readme.md")]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]

use rug::Float;
use std::f64;

fn zeta_f64(n: usize) -> f64 {
    // find number such that N^-n < ε ⇒ N > ε^-n
    let terms = (f64::EPSILON.powf(-1.0 / (n as f64)).ceil() as usize).max(1);
    (1..terms).map(|i| (i as f64).powi(-(n as i32))).sum()
}

fn k0_f64() -> f64 {
    let mut s1 = 0.0;
    for n in 1.. {
        let mut s2 = 0.0;
        for k in 1..2 * n {
            if k % 2 == 1 {
                s2 += 1.0 / (k as f64);
            } else {
                s2 -= 1.0 / (k as f64);
            }
        }
        s2 *= (zeta_f64(2 * n) - 1.0) / (n as f64);
        s1 += s2;
        if s2 < f64::EPSILON * s1 {
            break;
        }
    }
    (s1 / f64::consts::LN_2).exp()
}

fn k0(precision: u32) -> Float {
    let mut s1 = Float::with_val(precision, 0.0);
    for n in 1.. {
        let mut s2 = Float::with_val(precision, 0.0);
        for k in 1..2 * n {
            let kr = Float::with_val(precision, k).recip();
            if k % 2 == 1 {
                s2 += kr;
            } else {
                s2 -= kr;
            }
        }
        let s1_prev = s1.clone();
        s1 += s2 * (Float::with_val(precision, Float::zeta_u(2 * n)) - 1) / n;
        if s1 == s1_prev {
            break;
        }
    }
    (s1 / Float::with_val(precision, Float::ln_u(2))).exp()
}

fn main() {
    println!("{}", k0_f64());
    for i in 1.. {
        let n = i * 512;
        println!("{} {}", n, k0(n));
    }
}
