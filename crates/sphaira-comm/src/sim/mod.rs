//! [`SimComm`]: P simulated ranks as threads of one process (data-structures §7.1).

use std::any::Any;
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::comm::{Comm, check_allgatherv, check_alltoallv, displacements};
use crate::element::{Element, decode_into, encode, ordered_sum_words};

#[cfg(test)]
mod tests;

/// Identifies the collective a rank is executing, compared across ranks.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Tag {
    op: &'static str,
    root: Option<usize>,
    elem: &'static str,
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(root={:?}, elem={})", self.op, self.root, self.elem)
    }
}

/// What one rank deposits in a collective round.
struct Contribution {
    tag: Tag,
    meta: Vec<i64>,
    data: Vec<u64>,
}

struct Round {
    contribs: Vec<Option<Arc<Contribution>>>,
    arrived: usize,
    taken: usize,
}

struct State {
    /// Open collective rounds keyed by `(communicator id, call sequence number)`.
    rounds: HashMap<(u64, u64), Round>,
    /// `exited[g]`: global rank `g` has returned from (or panicked out of) its closure.
    exited: Vec<bool>,
    /// Global rank of the first rank that panicked, if any.
    first_panic: Option<usize>,
    /// Communicator ids handed out by `split`, keyed by `(parent id, split call #, color)`.
    comm_ids: HashMap<(u64, u64, u32), u64>,
    next_comm_id: u64,
}

struct World {
    state: Mutex<State>,
    cv: Condvar,
}

impl World {
    fn lock(&self) -> MutexGuard<'_, State> {
        // INVARIANT: no code panics while holding the lock in a way that leaves `State`
        // inconsistent, so a poisoned lock is still usable (rank panics are handled via `first_panic`).
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// `P` ranks simulated as threads of one process with shared mailboxes (data-structures §7.1).
///
/// Use [`SimComm::run`] to execute a closure on every rank. This lets ordinary `cargo test`
/// exercise partitioning and communication plans without `mpirun`.
///
/// Safety nets, so that tests fail loudly instead of hanging:
///
/// * every collective call carries a per-communicator sequence number and an operation tag
///   (operation, root, element type); ranks whose `n`-th calls differ cause a panic naming both
///   calls, and so does a rank that returns from its closure while others still wait for it;
/// * a panic in one rank wakes all waiting ranks, which panic in turn; [`SimComm::run`] then
///   re-raises the original panic.
///
/// A true circular wait across *different* communicators is not detected.
///
/// ```
/// use sphaira_comm::{Comm, SimComm};
/// let sums = SimComm::run(3, |c| {
///     let mut s = [0.0];
///     c.allreduce_ordered(&[c.rank() as f64], &mut s);
///     s[0]
/// });
/// assert_eq!(sums, vec![3.0, 3.0, 3.0]);
/// ```
pub struct SimComm {
    world: Arc<World>,
    id: u64,
    rank: usize,
    /// Global rank of each member of this communicator, indexed by rank in it.
    members: Arc<Vec<usize>>,
    seq: Cell<u64>,
}

impl fmt::Debug for SimComm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimComm")
            .field("id", &self.id)
            .field("rank", &self.rank)
            .field("size", &self.members.len())
            .finish()
    }
}

impl SimComm {
    /// Runs `f` on `p` simulated ranks (one thread each) and returns the results in rank order.
    ///
    /// Each rank receives its own communicator with `rank() == 0..p`.
    ///
    /// # Panics
    ///
    /// If `p == 0`. If any rank panics, the other ranks are woken and abort their pending
    /// collectives, and this function resumes the *first* panic (its original payload).
    pub fn run<R, F>(p: usize, f: F) -> Vec<R>
    where
        R: Send,
        F: Fn(&SimComm) -> R + Sync,
    {
        assert!(p >= 1, "SimComm::run needs at least one rank");
        let world = Arc::new(World {
            state: Mutex::new(State {
                rounds: HashMap::new(),
                exited: vec![false; p],
                first_panic: None,
                comm_ids: HashMap::new(),
                next_comm_id: 1,
            }),
            cv: Condvar::new(),
        });
        let members: Arc<Vec<usize>> = Arc::new((0..p).collect());
        let results: Vec<Result<R, Box<dyn Any + Send>>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..p)
                .map(|g| {
                    let comm = SimComm {
                        world: Arc::clone(&world),
                        id: 0,
                        rank: g,
                        members: Arc::clone(&members),
                        seq: Cell::new(0),
                    };
                    let f = &f;
                    s.spawn(move || {
                        let r = catch_unwind(AssertUnwindSafe(|| f(&comm)));
                        let mut st = comm.world.lock();
                        if r.is_err() && st.first_panic.is_none() {
                            st.first_panic = Some(g);
                        }
                        st.exited[g] = true;
                        drop(st);
                        comm.world.cv.notify_all();
                        r
                    })
                })
                .collect();
            handles
                .into_iter()
                // INVARIANT: the thread body catches every panic of `f`, so `join` cannot fail.
                .map(|h| {
                    h.join()
                        .expect("rank thread cannot panic outside catch_unwind")
                })
                .collect()
        });
        let first = world.lock().first_panic;
        let mut out = Vec::with_capacity(p);
        let mut payload = None;
        for (g, r) in results.into_iter().enumerate() {
            match r {
                Ok(v) => out.push(v),
                Err(e) if Some(g) == first => payload = Some(e),
                Err(_) => {}
            }
        }
        if let Some(e) = payload {
            resume_unwind(e);
        }
        out
    }

    /// Deposits this rank's contribution to the next collective round of this communicator and
    /// waits for all members. Returns the call sequence number and all contributions in rank order.
    fn exchange(&self, tag: Tag, meta: Vec<i64>, data: Vec<u64>) -> (u64, Vec<Arc<Contribution>>) {
        let seq = self.seq.get();
        self.seq.set(seq + 1);
        let size = self.members.len();
        let key = (self.id, seq);
        let mine = tag.clone();
        let mut st = self.world.lock();
        let round = st.rounds.entry(key).or_insert_with(|| Round {
            contribs: vec![None; size],
            arrived: 0,
            taken: 0,
        });
        round.contribs[self.rank] = Some(Arc::new(Contribution { tag, meta, data }));
        round.arrived += 1;
        self.world.cv.notify_all();
        loop {
            if let Some(g) = st.first_panic {
                drop(st);
                panic!(
                    "SimComm rank {} (comm {}, call #{seq}, {mine}): aborting because global rank {g} panicked",
                    self.rank, self.id
                );
            }
            let round = &st.rounds[&key];
            if round.arrived == size {
                break;
            }
            let gone =
                (0..size).find(|&m| round.contribs[m].is_none() && st.exited[self.members[m]]);
            if let Some(m) = gone {
                drop(st);
                panic!(
                    "SimComm collective mismatch on comm {} at call #{seq}: rank {} called {mine} \
                     but rank {m} already returned from its closure without making this call",
                    self.id, self.rank
                );
            }
            st = self.world.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        // INVARIANT: `arrived == size`, so every slot of `contribs` is `Some`.
        let round = st
            .rounds
            .get_mut(&key)
            .expect("round exists while members wait");
        let all: Vec<_> = round
            .contribs
            .iter()
            .map(|c| Arc::clone(c.as_ref().expect("all contributions present")))
            .collect();
        round.taken += 1;
        if round.taken == size {
            st.rounds.remove(&key);
        }
        drop(st);
        for (r, c) in all.iter().enumerate() {
            if c.tag != mine {
                panic!(
                    "SimComm collective mismatch on comm {} at call #{seq}: rank {} called {mine} \
                     but rank {r} called {}",
                    self.id, self.rank, c.tag
                );
            }
        }
        (seq, all)
    }

    fn tag<T: Element>(op: &'static str, root: Option<usize>) -> Tag {
        Tag {
            op,
            root,
            elem: T::NAME,
        }
    }

    fn check_root(&self, op: &str, root: usize) {
        assert!(
            root < self.size(),
            "{op}: root {root} out of range for size {}",
            self.size()
        );
    }

    /// Gathers equal-length contributions and checks the lengths agree; returns `(n, words)`.
    fn gather_equal<T: Element>(&self, tag: Tag, send: &[T]) -> Vec<Arc<Contribution>> {
        let (seq, all) = self.exchange(tag, vec![send.len() as i64], encode(send));
        for (r, c) in all.iter().enumerate() {
            assert_eq!(
                c.meta[0] as usize,
                send.len(),
                "SimComm length mismatch on comm {} at call #{seq}: rank {} sent {} elements, rank {r} sent {}",
                self.id,
                self.rank,
                send.len(),
                c.meta[0]
            );
        }
        all
    }
}

fn concat_words(all: &[Arc<Contribution>]) -> Vec<u64> {
    all.iter().flat_map(|c| c.data.iter().copied()).collect()
}

impl Comm for SimComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.members.len()
    }

    fn split(&self, color: u32, key: i32) -> Self {
        let (seq, all) = self.exchange(
            Tag {
                op: "split",
                root: None,
                elem: "-",
            },
            vec![i64::from(color), i64::from(key)],
            Vec::new(),
        );
        let mut group: Vec<(i64, usize)> = all
            .iter()
            .enumerate()
            .filter(|(_, c)| c.meta[0] == i64::from(color))
            .map(|(r, c)| (c.meta[1], r))
            .collect();
        group.sort_unstable();
        let rank = group
            .iter()
            .position(|&(_, r)| r == self.rank)
            // INVARIANT: this rank contributed `color`, so it is in its own group.
            .expect("own rank is in its color group");
        let members: Vec<usize> = group.iter().map(|&(_, r)| self.members[r]).collect();
        let mut st = self.world.lock();
        let next = st.next_comm_id;
        let id = *st.comm_ids.entry((self.id, seq, color)).or_insert(next);
        if id == next {
            st.next_comm_id += 1;
        }
        drop(st);
        SimComm {
            world: Arc::clone(&self.world),
            id,
            rank,
            members: Arc::new(members),
            seq: Cell::new(0),
        }
    }

    fn barrier(&self) {
        self.exchange(
            Tag {
                op: "barrier",
                root: None,
                elem: "-",
            },
            Vec::new(),
            Vec::new(),
        );
    }

    fn bcast<T: Element>(&self, root: usize, buf: &mut [T]) {
        self.check_root("bcast", root);
        let data = if self.rank == root {
            encode(buf)
        } else {
            Vec::new()
        };
        let (seq, all) = self.exchange(
            Self::tag::<T>("bcast", Some(root)),
            vec![buf.len() as i64],
            data,
        );
        let rc = &all[root];
        assert_eq!(
            rc.meta[0] as usize,
            buf.len(),
            "SimComm bcast length mismatch on comm {} at call #{seq}: rank {} has {} elements, root {root} has {}",
            self.id,
            self.rank,
            buf.len(),
            rc.meta[0]
        );
        if self.rank != root {
            decode_into(&rc.data, buf);
        }
    }

    fn allgatherv<T: Element>(&self, send: &[T], counts: &[usize], recv: &mut [T]) {
        check_allgatherv(self.rank, self.size(), send.len(), counts, recv.len());
        let (seq, all) = self.exchange(
            Self::tag::<T>("allgatherv", None),
            vec![send.len() as i64],
            encode(send),
        );
        for (r, c) in all.iter().enumerate() {
            assert_eq!(
                c.meta[0] as usize, counts[r],
                "SimComm allgatherv counts mismatch on comm {} at call #{seq}: rank {} expects {} \
                 elements from rank {r}, which sent {}",
                self.id, self.rank, counts[r], c.meta[0]
            );
        }
        decode_into(&concat_words(&all), recv);
    }

    fn alltoallv<T: Element>(
        &self,
        send: &[T],
        send_counts: &[usize],
        recv: &mut [T],
        recv_counts: &[usize],
    ) {
        let p = self.size();
        check_alltoallv(p, send.len(), send_counts, recv.len(), recv_counts);
        let meta: Vec<i64> = send_counts.iter().map(|&c| c as i64).collect();
        let (seq, all) = self.exchange(Self::tag::<T>("alltoallv", None), meta, encode(send));
        let mut off = 0usize;
        for (s, c) in all.iter().enumerate() {
            let sent = c.meta[self.rank] as usize;
            assert_eq!(
                sent, recv_counts[s],
                "SimComm alltoallv counts mismatch on comm {} at call #{seq}: rank {s} sends {sent} \
                 elements to rank {}, which expects {}",
                self.id, self.rank, recv_counts[s]
            );
            let (displ, _) = displacements(&c.meta.iter().map(|&x| x as usize).collect::<Vec<_>>());
            let start = displ[self.rank] * T::WORDS;
            let words = &c.data[start..start + sent * T::WORDS];
            decode_into(words, &mut recv[off..off + sent]);
            off += sent;
        }
    }

    fn reduce_ordered<T: Element>(&self, root: usize, send: &[T], recv: &mut [T]) {
        self.check_root("reduce_ordered", root);
        let all = self.gather_equal(Self::tag::<T>("reduce_ordered", Some(root)), send);
        if self.rank == root {
            assert_eq!(
                recv.len(),
                send.len(),
                "reduce_ordered: recv.len() != send.len() on root"
            );
            ordered_sum_words(&concat_words(&all), self.size(), recv);
        }
    }

    fn allreduce_ordered<T: Element>(&self, send: &[T], recv: &mut [T]) {
        assert_eq!(
            recv.len(),
            send.len(),
            "allreduce_ordered: recv.len() != send.len()"
        );
        let all = self.gather_equal(Self::tag::<T>("allreduce_ordered", None), send);
        ordered_sum_words(&concat_words(&all), self.size(), recv);
    }
}
