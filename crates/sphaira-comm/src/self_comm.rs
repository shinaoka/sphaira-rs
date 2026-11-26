//! [`SelfComm`]: one rank, no communication (data-structures §7.1).

use crate::comm::{Comm, check_allgatherv, check_alltoallv};
use crate::element::Element;

/// A communicator with exactly one rank (data-structures §7.1). No messages are ever sent;
/// every collective is a local copy.
///
/// ```
/// use sphaira_comm::{Comm, SelfComm};
/// let c = SelfComm;
/// let mut out = [0.0; 2];
/// c.allreduce_ordered(&[1.0, 2.0], &mut out);
/// assert_eq!(out, [1.0, 2.0]);
/// assert_eq!((c.rank(), c.size()), (0, 1));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct SelfComm;

impl Comm for SelfComm {
    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn split(&self, _color: u32, _key: i32) -> Self {
        SelfComm
    }

    fn barrier(&self) {}

    fn bcast<T: Element>(&self, root: usize, _buf: &mut [T]) {
        assert_eq!(root, 0, "bcast: root {root} out of range for size 1");
    }

    fn allgatherv<T: Element>(&self, send: &[T], counts: &[usize], recv: &mut [T]) {
        check_allgatherv(0, 1, send.len(), counts, recv.len());
        recv.copy_from_slice(send);
    }

    fn alltoallv<T: Element>(
        &self,
        send: &[T],
        send_counts: &[usize],
        recv: &mut [T],
        recv_counts: &[usize],
    ) {
        check_alltoallv(1, send.len(), send_counts, recv.len(), recv_counts);
        assert_eq!(
            send_counts[0], recv_counts[0],
            "alltoallv: recv_counts[0] must equal send_counts[0] on one rank"
        );
        recv.copy_from_slice(send);
    }

    fn reduce_ordered<T: Element>(&self, root: usize, send: &[T], recv: &mut [T]) {
        assert_eq!(
            root, 0,
            "reduce_ordered: root {root} out of range for size 1"
        );
        assert_eq!(
            send.len(),
            recv.len(),
            "reduce_ordered: recv.len() != send.len()"
        );
        recv.copy_from_slice(send);
    }

    fn allreduce_ordered<T: Element>(&self, send: &[T], recv: &mut [T]) {
        assert_eq!(
            send.len(),
            recv.len(),
            "allreduce_ordered: recv.len() != send.len()"
        );
        recv.copy_from_slice(send);
    }
}
