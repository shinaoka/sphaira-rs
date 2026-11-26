//! Checks every `MpiComm` collective against its definition.
//!
//! Run with any number of ranks, for example
//! `mpirun -n 3 target/debug/examples/mpi_check` after
//! `cargo build -p sphaira-comm --features mpi --example mpi_check`.

use sphaira_comm::{Comm, MpiComm};

fn main() {
    let universe = mpi::initialize().expect("MPI initialization failed");
    let comm = MpiComm::from_communicator(universe.world());
    let (me, p) = (comm.rank(), comm.size());

    comm.barrier();

    let mut b = [0i64; 2];
    if me == p - 1 {
        b = [5, -6];
    }
    comm.bcast(p - 1, &mut b);
    assert_eq!(b, [5, -6]);

    // Odd ranks contribute nothing.
    let send: Vec<f64> = if me % 2 == 1 {
        vec![]
    } else {
        vec![me as f64; 2]
    };
    let counts = comm.allgather_count(send.len());
    let mut g = vec![0.0; counts.iter().sum()];
    comm.allgatherv(&send, &counts, &mut g);
    let want: Vec<f64> = (0..p)
        .filter(|r| r % 2 == 0)
        .flat_map(|r| [r as f64; 2])
        .collect();
    assert_eq!(g, want);

    let send_counts: Vec<usize> = (0..p).map(|d| (me + 2 * d) % 3).collect();
    let send: Vec<usize> = (0..p)
        .flat_map(|d| vec![me * 100 + d; (me + 2 * d) % 3])
        .collect();
    let recv_counts = comm.alltoall_counts(&send_counts);
    let mut recv = vec![0usize; recv_counts.iter().sum()];
    comm.alltoallv(&send, &send_counts, &mut recv, &recv_counts);
    let want: Vec<usize> = (0..p)
        .flat_map(|s| vec![s * 100 + me; (s + 2 * me) % 3])
        .collect();
    assert_eq!(recv, want);

    let part = |r: usize| 0.1 * (r as f64 + 1.0).powi(3) + 1e16 * (r % 2) as f64;
    let mut serial = part(0);
    for r in 1..p {
        serial += part(r);
    }
    let mut out = [0.0];
    comm.allreduce_ordered(&[part(me)], &mut out);
    assert_eq!(out[0].to_bits(), serial.to_bits());
    let mut root_out = [0.0];
    comm.reduce_ordered(0, &[part(me)], &mut root_out);
    if me == 0 {
        assert_eq!(root_out[0].to_bits(), serial.to_bits());
    }

    let sub = comm.split((me % 2) as u32, me as i32);
    let mut n = [0usize];
    sub.allreduce_ordered(&[1], &mut n);
    assert_eq!(n[0], (0..p).filter(|r| r % 2 == me % 2).count());

    if me == 0 {
        println!("mpi_check: all collectives OK on {p} ranks");
    }
}
