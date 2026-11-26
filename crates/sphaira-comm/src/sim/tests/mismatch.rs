use crate::{Comm, SimComm};

#[test]
#[should_panic(expected = "collective mismatch")]
fn different_operations_panic() {
    SimComm::run(3, |c| {
        if c.rank() == 1 {
            let mut o = [0.0];
            c.allreduce_ordered(&[1.0], &mut o);
        } else {
            c.barrier();
        }
    });
}

#[test]
#[should_panic(expected = "collective mismatch")]
fn different_roots_panic() {
    SimComm::run(2, |c| {
        let mut b = [0.0];
        c.bcast(c.rank(), &mut b);
    });
}

#[test]
#[should_panic(expected = "collective mismatch")]
fn different_element_types_panic() {
    SimComm::run(2, |c| {
        if c.rank() == 0 {
            c.bcast(0, &mut [1.0f64]);
        } else {
            c.bcast(0, &mut [1i64]);
        }
    });
}

#[test]
#[should_panic(expected = "without making this call")]
fn rank_that_skips_a_collective_panics() {
    SimComm::run(3, |c| {
        if c.rank() != 2 {
            c.barrier();
        }
    });
}

#[test]
#[should_panic(expected = "length mismatch")]
fn unequal_reduction_lengths_panic() {
    SimComm::run(2, |c| {
        let send = vec![1.0; c.rank() + 1];
        let mut out = vec![0.0; c.rank() + 1];
        c.allreduce_ordered(&send, &mut out);
    });
}

#[test]
#[should_panic(expected = "counts mismatch")]
fn inconsistent_alltoallv_counts_panic() {
    SimComm::run(2, |c| {
        // Rank 0 sends one element to rank 1 but rank 1 expects none.
        let (send_counts, recv_counts, send_len) = if c.rank() == 0 {
            ([0, 1], [0, 0], 1)
        } else {
            ([0, 0], [0, 0], 0)
        };
        let mut recv: [f64; 0] = [];
        c.alltoallv(&vec![1.0; send_len], &send_counts, &mut recv, &recv_counts);
    });
}

#[test]
fn panic_in_one_rank_does_not_deadlock_and_keeps_original_message() {
    let r = std::panic::catch_unwind(|| {
        SimComm::run(4, |c| {
            if c.rank() == 2 {
                panic!("boom from rank 2");
            }
            c.barrier();
        })
    });
    let msg = *r.unwrap_err().downcast::<&str>().unwrap();
    assert_eq!(msg, "boom from rank 2");
}

#[test]
fn results_are_returned_in_rank_order() {
    let r = SimComm::run(5, |c| c.rank() * c.rank());
    assert_eq!(r, vec![0, 1, 4, 9, 16]);
}
