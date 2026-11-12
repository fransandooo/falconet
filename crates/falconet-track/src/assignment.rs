//! Optimal one-to-one assignment (the "linear assignment problem").
//!
//! Given a cost matrix (rows = tracks, columns = detections), find the
//! pairing with the smallest total cost, never pairing anything whose cost is
//! above `cost_limit`.
//!
//! ByteTrack calls `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`.
//! That extends the `n × m` matrix to a square `(n + m) × (n + m)` one where
//! every row and column also has a "dummy" partner costing `cost_limit / 2`:
//!
//! ```text
//!            m real cols     n dummy cols
//!          ┌──────────────┬──────────────┐
//! n real   │   cost[i][j] │  limit / 2   │
//! rows     ├──────────────┼──────────────┤
//! m dummy  │   limit / 2  │      0       │
//! rows     └──────────────┴──────────────┘
//! ```
//!
//! Pairing real row `i` with real column `j` costs `cost[i][j]`; leaving both
//! unmatched costs `limit / 2 + limit / 2 = limit`. So the optimum pairs them
//! only when `cost[i][j] < limit`. We do the same and solve the square problem
//! with the Hungarian algorithm (Kuhn–Munkres, O(N³)).

/// Result of an assignment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Assignment {
    /// `(row, col)` pairs, sorted by row.
    pub matches: Vec<(usize, usize)>,
    /// Rows left unmatched, ascending.
    pub unmatched_rows: Vec<usize>,
    /// Columns left unmatched, ascending.
    pub unmatched_cols: Vec<usize>,
}

/// Solves the assignment problem for a row-major `rows × cols` cost matrix.
pub fn linear_assignment(cost: &[f32], rows: usize, cols: usize, cost_limit: f32) -> Assignment {
    assert_eq!(cost.len(), rows * cols, "cost matrix must be rows × cols");
    if rows == 0 || cols == 0 {
        return Assignment {
            matches: Vec::new(),
            unmatched_rows: (0..rows).collect(),
            unmatched_cols: (0..cols).collect(),
        };
    }

    let n = rows + cols;
    let half = f64::from(cost_limit) / 2.0;
    let mut extended = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            extended[i * n + j] = match (i < rows, j < cols) {
                (true, true) => f64::from(cost[i * cols + j]),
                (false, false) => 0.0,
                _ => half,
            };
        }
    }
    let row_to_col = hungarian(&extended, n);

    let mut result = Assignment::default();
    let mut col_used = vec![false; cols];
    for (i, &j) in row_to_col.iter().enumerate().take(rows) {
        if j < cols {
            result.matches.push((i, j));
            col_used[j] = true;
        } else {
            result.unmatched_rows.push(i);
        }
    }
    result.unmatched_cols = (0..cols).filter(|&j| !col_used[j]).collect();
    result
}

/// Hungarian algorithm on a square `n × n` row-major matrix. Returns, for
/// each row, the column assigned to it.
///
/// This is the classic O(n³) shortest-augmenting-path formulation with dual
/// potentials `u` (rows) and `v` (columns). Index 0 is a sentinel, so rows
/// and columns are 1-based inside the loop.
fn hungarian(cost: &[f64], n: usize) -> Vec<usize> {
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    // p[j] = row matched to column j (0 = none); way[j] = previous column on
    // the augmenting path.
    let (mut p, mut way) = (vec![0_usize; n + 1], vec![0_usize; n + 1]);

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut min_slack = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let reduced = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if reduced < min_slack[j] {
                    min_slack[j] = reduced;
                    way[j] = j0;
                }
                if min_slack[j] < delta {
                    delta = min_slack[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_slack[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Flip the augmenting path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut row_to_col = vec![0; n];
    for j in 1..=n {
        if p[j] != 0 {
            row_to_col[p[j] - 1] = j - 1;
        }
    }
    row_to_col
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(cost: &[f32], cols: usize, a: &Assignment) -> f32 {
        a.matches.iter().map(|&(i, j)| cost[i * cols + j]).sum()
    }

    #[test]
    fn picks_the_global_optimum_not_the_greedy_one() {
        // Greedy would take (0,0)=0.1 and then be forced into (1,1)=0.9.
        // The optimum is (0,1) + (1,0) = 0.2 + 0.3 = 0.5 < 1.0.
        let cost = [0.1, 0.2, 0.3, 0.9];
        let a = linear_assignment(&cost, 2, 2, 1.0);
        assert_eq!(a.matches, vec![(0, 1), (1, 0)]);
        assert!((total(&cost, 2, &a) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pairs_above_the_limit_stay_unmatched() {
        let cost = [0.9, 0.2];
        let a = linear_assignment(&cost, 1, 2, 0.5);
        assert_eq!(a.matches, vec![(0, 1)]);
        assert_eq!(a.unmatched_cols, vec![0]);

        let a = linear_assignment(&[0.9], 1, 1, 0.5);
        assert!(a.matches.is_empty());
        assert_eq!((a.unmatched_rows, a.unmatched_cols), (vec![0], vec![0]));
    }

    #[test]
    fn rectangular_and_empty_matrices() {
        // 3 tracks, 1 detection.
        let a = linear_assignment(&[0.4, 0.1, 0.3], 3, 1, 0.8);
        assert_eq!(a.matches, vec![(1, 0)]);
        assert_eq!(a.unmatched_rows, vec![0, 2]);
        let a = linear_assignment(&[], 0, 4, 0.8);
        assert_eq!(a.unmatched_cols, vec![0, 1, 2, 3]);
    }

    #[test]
    fn matches_brute_force_on_random_matrices() {
        // Every permutation of 4 columns, compared against the solver.
        let mut seed = 1_u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let perms = permutations(4);
        for _ in 0..200 {
            let cost: Vec<f32> = (0..16).map(|_| rand()).collect();
            let best = perms
                .iter()
                .map(|p| {
                    p.iter()
                        .enumerate()
                        .map(|(i, &j)| cost[i * 4 + j])
                        .sum::<f32>()
                })
                .fold(f32::INFINITY, f32::min);
            // A limit above every cost forces a complete assignment.
            let a = linear_assignment(&cost, 4, 4, 2.0);
            assert_eq!(a.matches.len(), 4);
            assert!((total(&cost, 4, &a) - best).abs() < 1e-5);
        }
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut out = Vec::new();
        for p in permutations(n - 1) {
            for pos in 0..=p.len() {
                let mut q = p.clone();
                q.insert(pos, n - 1);
                out.push(q);
            }
        }
        out
    }
}
