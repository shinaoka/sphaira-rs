use super::PS;
use crate::{Comm, SelfComm, SimComm};

#[test]
fn split_by_parity_orders_by_key() {
    for p in PS {
        // Even ranks get descending keys, so their order inside the new communicator is reversed.
        let r = SimComm::run(p, |c| {
            let color = (c.rank() % 2) as u32;
            let key = if color == 0 {
                -(c.rank() as i32)
            } else {
                c.rank() as i32
            };
            let sub = c.split(color, key);
            let mut ranks = vec![0usize; sub.size()];
            sub.allgatherv(&[c.rank()], &vec![1; sub.size()], &mut ranks);
            let mut sum = [0usize];
            sub.allreduce_ordered(&[c.rank() + 1], &mut sum);
            (sub.rank(), sub.size(), ranks, sum[0])
        });
        for (g, (sr, ss, ranks, sum)) in r.into_iter().enumerate() {
            let mut members: Vec<usize> = (0..p).filter(|q| q % 2 == g % 2).collect();
            if g % 2 == 0 {
                members.reverse();
            }
            assert_eq!(ss, members.len());
            assert_eq!(members[sr], g);
            assert_eq!(ranks, members);
            assert_eq!(sum, members.iter().map(|m| m + 1).sum::<usize>());
        }
    }
}

#[test]
fn sub_and_parent_collectives_are_independent_and_nest() {
    let r = SimComm::run(5, |c| {
        let sub = c.split((c.rank() / 2) as u32, 0); // groups {0,1} {2,3} {4}
        let subsub = sub.split(0, -(sub.rank() as i32)); // reversed order in each group
        let mut total = [0.0];
        c.allreduce_ordered(&[1.0], &mut total);
        let mut group = [0.0];
        sub.allreduce_ordered(&[1.0], &mut group);
        sub.barrier();
        let mut b = [c.rank() as i64];
        subsub.bcast(0, &mut b);
        (total[0], group[0], subsub.rank(), b[0])
    });
    assert_eq!(r[0], (5.0, 2.0, 1, 1)); // reversed: rank 1 of the group is first, root value 1
    assert_eq!(r[1], (5.0, 2.0, 0, 1));
    assert_eq!(r[2], (5.0, 2.0, 1, 3));
    assert_eq!(r[3], (5.0, 2.0, 0, 3));
    assert_eq!(r[4], (5.0, 1.0, 0, 4));
}

#[test]
fn self_comm_split() {
    let s = SelfComm.split(3, 9);
    assert_eq!((s.rank(), s.size()), (0, 1));
}
