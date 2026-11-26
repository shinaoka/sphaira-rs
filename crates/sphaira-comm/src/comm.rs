//! The [`Comm`] trait (data-structures §7.1, §7.3).

use crate::element::Element;

/// Communicator: the only way drivers communicate (data-structures §2 item 4, §7.1).
///
/// All methods are *collective*: every rank of the communicator must call them in the same order
/// with consistent arguments. Implementations: [`crate::SelfComm`] (one rank),
/// [`crate::SimComm`] (ranks as threads of one process) and `MpiComm` (feature `mpi`).
///
/// # Buffer conventions
///
/// * Ranks are numbered `0..size()`.
/// * "Counts" are in elements, not bytes. The *displacement* of block `r` is the sum of the counts
///   of blocks `0..r` (blocks are packed back to back, in increasing rank order).
/// * Zero-length contributions are allowed everywhere.
/// * Argument or length inconsistencies are programming errors and panic (documented per method as
///   `# Panics`). [`crate::SimComm`] additionally detects calls that differ between ranks.
///
/// # Determinism (data-structures §7.3)
///
/// The reductions gather every rank's partial result and add them in increasing rank order, as
/// `((p0 + p1) + p2) + ...`, with `p0` taken unchanged. They never use an MPI reduction whose
/// order is unspecified, so with a fixed number of ranks the result is bitwise reproducible.
pub trait Comm {
    /// This rank's number in `0..size()`.
    fn rank(&self) -> usize;

    /// Number of ranks.
    fn size(&self) -> usize;

    /// Splits the communicator (collective on `self`).
    ///
    /// Ranks that pass the same `color` form one new communicator; inside it ranks are ordered by
    /// increasing `key`, ties broken by the rank in `self`. Every rank belongs to exactly one new
    /// communicator (there is no "undefined" color). Collectives on the result are independent of
    /// collectives on `self`.
    fn split(&self, color: u32, key: i32) -> Self
    where
        Self: Sized;

    /// Returns on every rank only after all ranks have entered it.
    fn barrier(&self);

    /// Broadcast from rank `root`: after the call, `buf` on every rank equals `root`'s `buf`.
    ///
    /// # Panics
    ///
    /// If `root >= size()` or `buf.len()` differs between ranks.
    fn bcast<T: Element>(&self, root: usize, buf: &mut [T]);

    /// Gathers every rank's `send` on every rank.
    ///
    /// `counts[r]` is the length of rank `r`'s `send`; every rank passes the same `counts`
    /// (`counts.len() == size()`, `counts[rank()] == send.len()`). On return
    /// `recv[displ(r)..displ(r) + counts[r]]` holds rank `r`'s data, where `displ(r)` is the sum of
    /// `counts[..r]`; `recv.len()` must equal the sum of `counts`.
    ///
    /// # Panics
    ///
    /// On any violation of the length rules above, including `counts` that disagree with the
    /// lengths the other ranks actually sent.
    fn allgatherv<T: Element>(&self, send: &[T], counts: &[usize], recv: &mut [T]);

    /// Personalized all-to-all exchange.
    ///
    /// `send` holds, back to back, the block for rank 0, rank 1, ...; block `d` has
    /// `send_counts[d]` elements (`send.len()` is their sum). On return `recv` holds, back to back,
    /// the block received from rank 0, rank 1, ...; block `s` has `recv_counts[s]` elements and is
    /// what rank `s` addressed to this rank. Consistency requirement: `recv_counts[s]` on this rank
    /// equals `send_counts[this rank]` on rank `s`. Both count slices have length `size()`.
    ///
    /// # Panics
    ///
    /// On any violation of the length or consistency rules above.
    fn alltoallv<T: Element>(
        &self,
        send: &[T],
        send_counts: &[usize],
        recv: &mut [T],
        recv_counts: &[usize],
    );

    /// Ordered element-wise sum onto rank `root`.
    ///
    /// Every rank passes a `send` of the same length `n`. On `root`, `recv` (length `n`) is
    /// overwritten with the rank-ordered sum; on all other ranks `recv` is left untouched (it may
    /// be empty).
    ///
    /// # Panics
    ///
    /// If `root >= size()`, the `send` lengths differ between ranks, or `root`'s `recv.len()`
    /// differs from `send.len()`.
    fn reduce_ordered<T: Element>(&self, root: usize, send: &[T], recv: &mut [T]);

    /// Ordered element-wise sum delivered to every rank.
    ///
    /// Like [`Comm::reduce_ordered`], but `recv` (length `n`) is overwritten on all ranks, with
    /// bitwise identical contents.
    ///
    /// # Panics
    ///
    /// If the `send` lengths differ between ranks or `recv.len() != send.len()`.
    fn allreduce_ordered<T: Element>(&self, send: &[T], recv: &mut [T]);

    /// Convenience: every rank's value `n` gathered on every rank (`result[r]` is rank `r`'s `n`).
    /// Typical use is to obtain the `counts` argument of [`Comm::allgatherv`].
    fn allgather_count(&self, n: usize) -> Vec<usize> {
        let mut out = vec![0usize; self.size()];
        let ones = vec![1usize; self.size()];
        self.allgatherv(&[n], &ones, &mut out);
        out
    }

    /// Convenience: given this rank's `send_counts` (length `size()`), returns the matching
    /// `recv_counts` for [`Comm::alltoallv`] (`result[s]` is what rank `s` sends to this rank).
    ///
    /// # Panics
    ///
    /// If `send_counts.len() != size()`.
    fn alltoall_counts(&self, send_counts: &[usize]) -> Vec<usize> {
        let p = self.size();
        assert_eq!(
            send_counts.len(),
            p,
            "send_counts must have one entry per rank"
        );
        let ones = vec![1usize; p];
        let mut recv_counts = vec![0usize; p];
        self.alltoallv(send_counts, &ones, &mut recv_counts, &ones);
        recv_counts
    }
}

/// Packed displacements (exclusive prefix sums) of `counts`, plus the total.
pub(crate) fn displacements(counts: &[usize]) -> (Vec<usize>, usize) {
    let mut total = 0usize;
    let d = counts
        .iter()
        .map(|&c| {
            let o = total;
            total += c;
            o
        })
        .collect();
    (d, total)
}

/// Local argument validation shared by all implementations of `allgatherv`.
pub(crate) fn check_allgatherv(
    rank: usize,
    size: usize,
    send: usize,
    counts: &[usize],
    recv: usize,
) {
    assert_eq!(
        counts.len(),
        size,
        "allgatherv: counts.len() must equal size()"
    );
    assert_eq!(
        counts[rank], send,
        "allgatherv: counts[rank] ({}) != send.len() ({send})",
        counts[rank]
    );
    let total: usize = counts.iter().sum();
    assert_eq!(
        recv, total,
        "allgatherv: recv.len() ({recv}) != sum of counts ({total})"
    );
}

/// Local argument validation shared by all implementations of `alltoallv`.
pub(crate) fn check_alltoallv(
    size: usize,
    send: usize,
    send_counts: &[usize],
    recv: usize,
    recv_counts: &[usize],
) {
    assert_eq!(
        send_counts.len(),
        size,
        "alltoallv: send_counts.len() must equal size()"
    );
    assert_eq!(
        recv_counts.len(),
        size,
        "alltoallv: recv_counts.len() must equal size()"
    );
    let st: usize = send_counts.iter().sum();
    let rt: usize = recv_counts.iter().sum();
    assert_eq!(
        send, st,
        "alltoallv: send.len() ({send}) != sum of send_counts ({st})"
    );
    assert_eq!(
        recv, rt,
        "alltoallv: recv.len() ({recv}) != sum of recv_counts ({rt})"
    );
}
