//! [`MpiComm`]: real MPI through rsmpi-rt (data-structures §7.1). Feature `mpi`.

use std::borrow::Cow;

use mpi::Count;
use mpi::collective::CommunicatorCollectives;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::{Color, SimpleCommunicator};
use mpi::traits::{Communicator, Root};

use crate::comm::{Comm, check_allgatherv, check_alltoallv, displacements};
use crate::element::{Element, ordered_sum_words};

/// A communicator backed by an MPI communicator (data-structures §7.1).
///
/// Elements travel as `u64` words (see [`Element`]); `f64`, `i64`, `u64` and `Complex<f64>`
/// buffers are sent without conversion. The ordered reductions never use `MPI_Reduce` or
/// `MPI_Allreduce` (data-structures §7.3): they gather the partial results and sum them in
/// increasing rank order. Unlike [`crate::SimComm`], mismatched collective sequences are *not*
/// detected; they behave as in MPI.
///
/// Obtain the MPI environment with `mpi::initialize()` (keep the `Universe` alive) and wrap
/// `universe.world()` with [`MpiComm::from_communicator`]. Feature `mpi` uses rsmpi-rt's
/// runtime-loaded backend (no MPI headers needed; set `MPI_RT_LIB` to an MPIwrapper library).
/// Check it with `cargo build -p sphaira-comm --features mpi --example mpi_check` and
/// `MPI_RT_LIB=/path/to/libmpiwrapper.so mpirun -n 3 target/debug/examples/mpi_check`
/// (see `examples/mpi_check.rs`).
pub struct MpiComm {
    comm: SimpleCommunicator,
}

impl std::fmt::Debug for MpiComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MpiComm")
            .field("rank", &self.comm.rank())
            .field("size", &self.comm.size())
            .finish()
    }
}

fn count(n: usize) -> Count {
    Count::try_from(n).expect("message length exceeds MPI count range")
}

fn counts_displs(counts: &[usize], words: usize) -> (Vec<Count>, Vec<Count>) {
    let (d, _) = displacements(counts);
    (
        counts.iter().map(|&c| count(c * words)).collect(),
        d.iter().map(|&x| count(x * words)).collect(),
    )
}

/// Runs `recv_words` with a word buffer for `out`: the elements themselves when they are plain
/// words, otherwise a temporary that is converted back afterwards.
fn with_recv_words<T: Element>(out: &mut [T], f: impl FnOnce(&mut [u64])) {
    if let Some(w) = T::as_words_mut(out) {
        f(w);
    } else {
        let mut tmp = vec![0u64; out.len() * T::WORDS];
        f(&mut tmp);
        T::from_words(&tmp, out);
    }
}

impl MpiComm {
    /// Wraps an existing `mpi` communicator (for example `universe.world()`).
    ///
    /// Together with [`MpiComm::communicator`] this lets other MPI users share one communicator:
    /// tenferro-mpi (point-to-point transfer of tenferro tensors) and pencil-rs (distributed FFT)
    /// take the same `mpi` crate's communicator, which is the only `mpi` crate in the build.
    pub fn from_communicator(comm: SimpleCommunicator) -> Self {
        Self { comm }
    }

    /// Borrows the underlying `mpi` communicator, for libraries that need MPI directly
    /// (tenferro-mpi, pencil-rs). Calls made on it interleave with `Comm` collectives in program
    /// order like any other MPI calls.
    pub fn communicator(&self) -> &SimpleCommunicator {
        &self.comm
    }

    /// Unwraps into the underlying `mpi` communicator.
    pub fn into_communicator(self) -> SimpleCommunicator {
        self.comm
    }

    /// Gathers equal-length blocks onto `root` and sums them there in rank order into `out`
    /// (only meaningful on `root`; other ranks pass `None`).
    fn gather_sum<T: Element>(&self, root: usize, send: &[T], out: Option<&mut [T]>) {
        let words: Cow<'_, [u64]> = T::as_words(send);
        let target = self.comm.process_at_rank(root as i32);
        match out {
            Some(out) => {
                let mut all = vec![0u64; words.len() * self.size()];
                target.gather_into_root(&words[..], &mut all[..]);
                ordered_sum_words(&all, self.size(), out);
            }
            None => target.gather_into(&words[..]),
        }
    }
}

impl Comm for MpiComm {
    fn rank(&self) -> usize {
        self.comm.rank() as usize
    }

    fn size(&self) -> usize {
        self.comm.size() as usize
    }

    fn split(&self, color: u32, key: i32) -> Self {
        let color = i32::try_from(color).expect("split: color must fit in i32");
        let comm = self
            .comm
            .split_by_color_with_key(Color::with_value(color), key)
            .expect("split with a defined color returns a communicator");
        Self { comm }
    }

    fn barrier(&self) {
        self.comm.barrier();
    }

    fn bcast<T: Element>(&self, root: usize, buf: &mut [T]) {
        assert!(root < self.size(), "bcast: root {root} out of range");
        let target = self.comm.process_at_rank(root as i32);
        if self.rank() == root {
            let mut w = T::as_words(buf).into_owned();
            target.broadcast_into(&mut w[..]);
        } else {
            with_recv_words(buf, |w| target.broadcast_into(w));
        }
    }

    fn allgatherv<T: Element>(&self, send: &[T], counts: &[usize], recv: &mut [T]) {
        check_allgatherv(self.rank(), self.size(), send.len(), counts, recv.len());
        let sw = T::as_words(send);
        let (c, d) = counts_displs(counts, T::WORDS);
        with_recv_words(recv, |w| {
            let mut part = PartitionMut::new(w, c, d);
            self.comm.all_gather_varcount_into(&sw[..], &mut part);
        });
    }

    fn alltoallv<T: Element>(
        &self,
        send: &[T],
        send_counts: &[usize],
        recv: &mut [T],
        recv_counts: &[usize],
    ) {
        check_alltoallv(
            self.size(),
            send.len(),
            send_counts,
            recv.len(),
            recv_counts,
        );
        let sw = T::as_words(send);
        let (sc, sd) = counts_displs(send_counts, T::WORDS);
        let (rc, rd) = counts_displs(recv_counts, T::WORDS);
        with_recv_words(recv, |w| {
            let sp = Partition::new(&sw[..], sc, sd);
            let mut rp = PartitionMut::new(w, rc, rd);
            self.comm.all_to_all_varcount_into(&sp, &mut rp);
        });
    }

    fn reduce_ordered<T: Element>(&self, root: usize, send: &[T], recv: &mut [T]) {
        assert!(
            root < self.size(),
            "reduce_ordered: root {root} out of range"
        );
        if self.rank() == root {
            assert_eq!(
                recv.len(),
                send.len(),
                "reduce_ordered: recv.len() != send.len() on root"
            );
            self.gather_sum(root, send, Some(recv));
        } else {
            self.gather_sum(root, send, None);
        }
    }

    fn allreduce_ordered<T: Element>(&self, send: &[T], recv: &mut [T]) {
        assert_eq!(
            recv.len(),
            send.len(),
            "allreduce_ordered: recv.len() != send.len()"
        );
        if self.rank() == 0 {
            self.gather_sum(0, send, Some(&mut *recv));
        } else {
            self.gather_sum(0, send, None);
        }
        self.bcast(0, recv);
    }
}
