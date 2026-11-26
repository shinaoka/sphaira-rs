//! [`MpiComm`]: real MPI through rsmpi-rt (data-structures §7.1). Feature `mpi`.

use mpi::Count;
use mpi::collective::CommunicatorCollectives;
use mpi::datatype::{Partition, PartitionMut};
use mpi::topology::{Color, SimpleCommunicator};
use mpi::traits::{Communicator, Root};

use crate::comm::{Comm, check_allgatherv, check_alltoallv, displacements};
use crate::element::{Element, decode_into, encode, ordered_sum_words};

/// A communicator backed by an MPI communicator (data-structures §7.1).
///
/// All elements travel as `u64` words (see [`Element`]), and the ordered reductions are built from
/// an all-gather followed by a local rank-ordered sum, never from `MPI_Reduce`/`MPI_Allreduce`
/// (data-structures §7.3). Unlike [`crate::SimComm`], mismatched collective sequences are *not*
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

    fn allgather_words(&self, send: &[u64], counts: &[usize], words: usize) -> Vec<u64> {
        let (c, d) = counts_displs(counts, words);
        let total: usize = counts.iter().sum::<usize>() * words;
        let mut out = vec![0u64; total];
        let mut part = PartitionMut::new(&mut out[..], c, d);
        self.comm.all_gather_varcount_into(send, &mut part);
        out
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
        let mut w = encode(buf);
        self.comm
            .process_at_rank(root as i32)
            .broadcast_into(&mut w[..]);
        decode_into(&w, buf);
    }

    fn allgatherv<T: Element>(&self, send: &[T], counts: &[usize], recv: &mut [T]) {
        check_allgatherv(self.rank(), self.size(), send.len(), counts, recv.len());
        let out = self.allgather_words(&encode(send), counts, T::WORDS);
        decode_into(&out, recv);
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
        let sw = encode(send);
        let mut rw = vec![0u64; recv.len() * T::WORDS];
        let (sc, sd) = counts_displs(send_counts, T::WORDS);
        let (rc, rd) = counts_displs(recv_counts, T::WORDS);
        let sp = Partition::new(&sw[..], sc, sd);
        let mut rp = PartitionMut::new(&mut rw[..], rc, rd);
        self.comm.all_to_all_varcount_into(&sp, &mut rp);
        decode_into(&rw, recv);
    }

    fn reduce_ordered<T: Element>(&self, root: usize, send: &[T], recv: &mut [T]) {
        assert!(
            root < self.size(),
            "reduce_ordered: root {root} out of range"
        );
        let n = send.len();
        let counts = vec![n; self.size()];
        let all = self.allgather_words(&encode(send), &counts, T::WORDS);
        if self.rank() == root {
            assert_eq!(
                recv.len(),
                n,
                "reduce_ordered: recv.len() != send.len() on root"
            );
            ordered_sum_words(&all, self.size(), recv);
        }
    }

    fn allreduce_ordered<T: Element>(&self, send: &[T], recv: &mut [T]) {
        assert_eq!(
            recv.len(),
            send.len(),
            "allreduce_ordered: recv.len() != send.len()"
        );
        let counts = vec![send.len(); self.size()];
        let all = self.allgather_words(&encode(send), &counts, T::WORDS);
        ordered_sum_words(&all, self.size(), recv);
    }
}
