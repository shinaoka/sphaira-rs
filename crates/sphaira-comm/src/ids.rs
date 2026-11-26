//! Index newtypes (data-structures §3 "Index spaces").
//!
//! Each index space is a distinct type so that, for example, an atom index cannot be passed where a
//! pair index is expected. The types are deliberately minimal: `Copy`, `Eq`, `Ord`, `Hash`, `Debug`,
//! a constructor and an accessor.

macro_rules! index_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(usize);

        impl $name {
            /// Wraps a raw index.
            pub const fn new(index: usize) -> Self {
                Self(index)
            }

            /// Returns the raw index.
            pub const fn index(self) -> usize {
                self.0
            }

            /// Alias of [`Self::index`].
            pub const fn get(self) -> usize {
                self.0
            }
        }

        impl From<usize> for $name {
            fn from(index: usize) -> Self {
                Self(index)
            }
        }
    };
}

index_newtype!(
    /// Species in the input order, range `0..n_species` (data-structures §3).
    SpeciesId
);
index_newtype!(
    /// Atom in the input order, range `0..n_atoms` (data-structures §3).
    AtomId
);
index_newtype!(
    /// Entry `(I, J, n)` of the pair graph, range `0..n_pairs` (data-structures §3, §4.1).
    PairId
);
index_newtype!(
    /// Global orbital index, atom-major and then in the order of the PAO basis chapter;
    /// range `0..n_orb` (data-structures §3).
    OrbId
);
index_newtype!(
    /// Linear row-major index `(i1, i2, i3)` of a grid point in the unit cell,
    /// range `0..N1*N2*N3` (data-structures §3).
    GridIndex
);
index_newtype!(
    /// Point of the irreducible k list, range `0..n_k` (data-structures §3).
    KIndex
);
index_newtype!(
    /// Spin channel, range `0..n_spin` with `n_spin` 1 or 2 (data-structures §3).
    Spin
);
index_newtype!(
    /// Pair `(KIndex, Spin)` flattened to one index, the unit of work of the eigenvalue stage;
    /// range `0..n_k*n_spin` (data-structures §3).
    KTask
);

/// Lattice translation `n` in Z^3 of the column-side atom (data-structures §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Cell([i32; 3]);

impl Cell {
    /// Wraps the integer translation vector.
    pub const fn new(n: [i32; 3]) -> Self {
        Self(n)
    }

    /// Returns the integer translation vector.
    pub const fn get(self) -> [i32; 3] {
        self.0
    }
}

impl From<[i32; 3]> for Cell {
    fn from(n: [i32; 3]) -> Self {
        Self(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_order() {
        assert_eq!(AtomId::new(3).index(), 3);
        assert_eq!(KTask::from(7).get(), 7);
        assert!(PairId::new(1) < PairId::new(2));
        assert_eq!(Cell::new([1, -2, 0]).get(), [1, -2, 0]);
        assert!(Cell::new([0, 0, 1]) > Cell::new([0, 0, 0]));
    }
}
