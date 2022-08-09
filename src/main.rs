#![doc = include_str!("../Readme.md")]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]

use rug::{float::Constant, Float};

fn k0(precision: u32) -> Float {
    // ζ(2) = π²/6
    let mut s1: Float = Float::with_val(precision, Constant::Pi).square() / 6 - 1;

    let mut s2: Float = Float::with_val(precision, 1);
    for n in 2_u32.. {
        let s1_prev = s1.clone();
        s2 -= Float::with_val(precision, (2 * n - 2) * (2 * n - 1)).recip();
        s1 += &s2 * (Float::with_val(precision, Float::zeta_u(2 * n)) - 1) / n;
        if s1 == s1_prev {
            break;
        }
    }
    (s1 / Float::with_val(precision, Constant::Log2)).exp()
}

fn main() {
    for i in 1.. {
        let n = i * 512;
        println!("{} {}", n, k0(n));
    }
}
