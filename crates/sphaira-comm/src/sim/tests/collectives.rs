use num_complex::Complex;

use super::PS;
use crate::{Comm, SelfComm, SimComm};

/// Length contributed by `rank`; rank 1 (when present) and every rank for odd P contribute nothing
/// in some tests, so zero-length data is covered.
fn len_of(rank: usize) -> usize {
    if rank % 2 == 1 { 0 } else { rank + 2 }
}

fn val(rank: usize, i: usize) -> f64 {
    (rank * 100 + i) as f64 + 0.25
}

#[test]
fn rank_size_barrier() {
    for p in PS {
        let r = SimComm::run(p, |c| {
            c.barrier();
            (c.rank(), c.size())
        });
        assert_eq!(r, (0..p).map(|i| (i, p)).collect::<Vec<_>>());
    }
}

#[test]
fn bcast_all_roots_and_empty() {
    for p in PS {
        for root in 0..p {
            let r = SimComm::run(p, |c| {
                let mut v = if c.rank() == root {
                    vec![1.5, -2.5, 3.5]
                } else {
                    vec![0.0; 3]
                };
                c.bcast(root, &mut v);
                let mut z = Complex::new(c.rank() as f64, 1.0);
                let mut zs = [z];
                c.bcast(root, &mut zs);
                z = zs[0];
                let mut empty: [i64; 0] = [];
                c.bcast(root, &mut empty);
                (v, z)
            });
            for (v, z) in r {
                assert_eq!(v, vec![1.5, -2.5, 3.5]);
                assert_eq!(z, Complex::new(root as f64, 1.0));
            }
        }
    }
}

#[test]
fn allgatherv_with_zero_length_ranks() {
    for p in PS {
        let r = SimComm::run(p, |c| {
            let send: Vec<f64> = (0..len_of(c.rank())).map(|i| val(c.rank(), i)).collect();
            let counts = c.allgather_count(send.len());
            let mut recv = vec![0.0; counts.iter().sum()];
            c.allgatherv(&send, &counts, &mut recv);
            (counts, recv)
        });
        let want_counts: Vec<usize> = (0..p).map(len_of).collect();
        let want: Vec<f64> = (0..p)
            .flat_map(|q| (0..len_of(q)).map(move |i| val(q, i)))
            .collect();
        for (counts, recv) in r {
            assert_eq!(counts, want_counts);
            assert_eq!(recv, want);
        }
    }
}

#[test]
fn allgatherv_all_element_types() {
    let r = SimComm::run(3, |c| {
        let me = c.rank();
        let mut a = [0i64; 3];
        c.allgatherv(&[-(me as i64) - 1], &[1, 1, 1], &mut a);
        let mut b = [0u64; 3];
        c.allgatherv(&[u64::MAX - me as u64], &[1, 1, 1], &mut b);
        let mut d = [0usize; 3];
        c.allgatherv(&[me * 7], &[1, 1, 1], &mut d);
        let mut z = [Complex::new(0.0, 0.0); 3];
        c.allgatherv(&[Complex::new(me as f64, -(me as f64))], &[1, 1, 1], &mut z);
        (a, b, d, z)
    });
    for (a, b, d, z) in r {
        assert_eq!(a, [-1, -2, -3]);
        assert_eq!(b, [u64::MAX, u64::MAX - 1, u64::MAX - 2]);
        assert_eq!(d, [0, 7, 14]);
        assert_eq!(z[2], Complex::new(2.0, -2.0));
    }
}

/// Number of elements rank `s` sends to rank `d`; some pairs are empty.
fn pair_len(s: usize, d: usize) -> usize {
    (s + 2 * d) % 3
}

#[test]
fn alltoallv_with_empty_blocks() {
    for p in PS {
        let r = SimComm::run(p, |c| {
            let me = c.rank();
            let send_counts: Vec<usize> = (0..p).map(|d| pair_len(me, d)).collect();
            let send: Vec<f64> = (0..p)
                .flat_map(|d| (0..pair_len(me, d)).map(move |i| val(me, d * 10 + i)))
                .collect();
            let recv_counts = c.alltoall_counts(&send_counts);
            let mut recv = vec![0.0; recv_counts.iter().sum()];
            c.alltoallv(&send, &send_counts, &mut recv, &recv_counts);
            (recv_counts, recv)
        });
        for (me, (recv_counts, recv)) in r.into_iter().enumerate() {
            let want_counts: Vec<usize> = (0..p).map(|s| pair_len(s, me)).collect();
            let want: Vec<f64> = (0..p)
                .flat_map(|s| (0..pair_len(s, me)).map(move |i| val(s, me * 10 + i)))
                .collect();
            assert_eq!(recv_counts, want_counts);
            assert_eq!(recv, want);
        }
    }
}

fn contribution(rank: usize, n: usize) -> Vec<f64> {
    // Values chosen so that floating-point addition is order dependent.
    (0..n)
        .map(|i| 0.1 * (rank as f64 + 1.0).powi(3) + 1e16 * ((rank + i) % 2) as f64 - 3e15)
        .collect()
}

fn serial_sum(p: usize, n: usize) -> Vec<f64> {
    let mut acc = contribution(0, n);
    for r in 1..p {
        for (a, x) in acc.iter_mut().zip(contribution(r, n)) {
            *a += x;
        }
    }
    acc
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn allreduce_ordered_equals_serial_and_is_repeatable() {
    for p in PS {
        let n = 7;
        let want = bits(&serial_sum(p, n));
        for _ in 0..10 {
            let r = SimComm::run(p, |c| {
                let mut out = vec![0.0; n];
                c.allreduce_ordered(&contribution(c.rank(), n), &mut out);
                bits(&out)
            });
            for got in r {
                assert_eq!(got, want);
            }
        }
    }
}

#[test]
fn reduce_ordered_root_only_and_empty() {
    for p in PS {
        let n = 4;
        let want = bits(&serial_sum(p, n));
        for root in 0..p {
            let r = SimComm::run(p, |c| {
                let mut out = if c.rank() == root {
                    vec![0.0; n]
                } else {
                    vec![-1.0; n]
                };
                c.reduce_ordered(root, &contribution(c.rank(), n), &mut out);
                let mut none: [f64; 0] = [];
                c.reduce_ordered(root, &[], &mut none);
                bits(&out)
            });
            for (me, got) in r.into_iter().enumerate() {
                if me == root {
                    assert_eq!(got, want);
                } else {
                    assert_eq!(got, bits(&vec![-1.0; n]), "non-root recv must be untouched");
                }
            }
        }
    }
}

#[test]
fn reductions_complex_and_integers() {
    for p in PS {
        let r = SimComm::run(p, |c| {
            let me = c.rank();
            let mut z = [Complex::new(0.0, 0.0); 2];
            c.allreduce_ordered(
                &[
                    Complex::new(me as f64, 1.0),
                    Complex::new(0.5, -(me as f64)),
                ],
                &mut z,
            );
            let mut i = [0i64; 1];
            c.allreduce_ordered(&[me as i64 - 1], &mut i);
            let mut u = [0usize; 1];
            c.allreduce_ordered(&[me + 1], &mut u);
            (z, i[0], u[0])
        });
        let sum: usize = (0..p).sum();
        for (z, i, u) in r {
            assert_eq!(z[0], Complex::new(sum as f64, p as f64));
            assert_eq!(z[1], Complex::new(0.5 * p as f64, -(sum as f64)));
            assert_eq!(i, sum as i64 - p as i64);
            assert_eq!(u, sum + p);
        }
    }
}

#[test]
fn self_comm_equals_sim_comm_p1() {
    fn script<C: Comm>(c: &C) -> Vec<u64> {
        let mut out = Vec::new();
        let mut b = [1.25, 2.5];
        c.bcast(0, &mut b);
        out.extend(bits(&b));
        let counts = c.allgather_count(3);
        out.extend(counts.iter().map(|&x| x as u64));
        let mut g = [0.0; 3];
        c.allgatherv(&[1.0, 2.0, 3.0], &counts, &mut g);
        out.extend(bits(&g));
        let rc = c.alltoall_counts(&[2]);
        let mut t = [0.0; 2];
        c.alltoallv(&[7.0, 8.0], &[2], &mut t, &rc);
        out.extend(bits(&t));
        let mut r = [0.0; 2];
        c.reduce_ordered(0, &[0.1, 0.2], &mut r);
        out.extend(bits(&r));
        let mut a = [0.0; 2];
        c.allreduce_ordered(&[0.3, 0.4], &mut a);
        out.extend(bits(&a));
        let s = c.split(4, 0);
        out.push((s.rank() * 10 + s.size()) as u64);
        c.barrier();
        out
    }
    let sim = SimComm::run(1, script);
    assert_eq!(sim[0], script(&SelfComm));
}
