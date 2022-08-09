#![doc = include_str!("../Readme.md")]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]

use std::f64;

fn zeta_f64(n: usize) -> f64 {
    let mut z = 1.0;
    for i in 2..1000 {
        z += 1.0 / (i as f64).powi(n as i32);
    }
    z
}

fn k0_f64(terms: usize) -> f64 {
    let mut s1 = 0.0;
    for n in 1..=terms {
        let mut s2 = 0.0;
        for k in 1..2 * n {
            if k % 2 == 1 {
                s2 += 1.0 / (k as f64);
            } else {
                s2 -= 1.0 / (k as f64);
            }
        }
        s1 += s2 * (zeta_f64(2 * n) - 1.0) / (n as f64);
    }
    (s1 / f64::consts::LN_2).exp()
}

fn main() {
    for i in 1..=100 {
        println!("{} {}", i, k0_f64(i));
    }
}
