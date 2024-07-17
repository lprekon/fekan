use rustc_hash::FxHashMap;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    usize,
};

#[derive(Debug)]
/// The number of cache hits and misses for a particular cache cell, where every B_i_k() basis function has a cell that caches input->output mappings
pub struct CacheStats {
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
pub(crate) struct BasisCache {
    /*                  k        k - 1     ...
     *                  |          |
     *              ---------  ---------  ----
     *             |         ||         ||
     * cache =  - [[CacheCell, CacheCell, ...],
     *         |   [CacheCell, CacheCell, ...],
     *         |   [CacheCell, CacheCell, ...],
     *  index -|   ...
     *         |   [CacheCell, CacheCell, ...],
     *         |   [CacheCell, CacheCell, ...],
     *          -  [CacheCell, CacheCell, ...]]
     *
     */
    cache: Vec<Vec<Mutex<FxHashMap<u64, f64>>>>,
    top_level_degree: usize,
    stats: Vec<Vec<CacheStats>>,
}

impl BasisCache {
    pub fn new(num_knots: usize, spline_degree: usize) -> Self {
        let mut cache = Vec::with_capacity(num_knots);
        let mut stats = Vec::with_capacity(num_knots);
        for _ in 0..num_knots {
            let mut cache_row = Vec::with_capacity(spline_degree);
            let mut stats_row = Vec::with_capacity(spline_degree);
            for _ in 0..spline_degree {
                cache_row.push(Mutex::new(FxHashMap::default()));
                stats_row.push(CacheStats::default());
            }
            cache.push(cache_row);
            stats.push(stats_row);
        }
        Self {
            cache,
            top_level_degree: spline_degree,
            stats,
        }
    }

    pub fn get(&self, i: usize, k: usize, key: u64) -> Option<f64> {
        let column_idx = self.top_level_degree - k;
        let cache_cell = &self.cache[i][column_idx];
        let cell_map = cache_cell.lock().unwrap();
        match cell_map.get(&key) {
            Some(value) => {
                self.stats[i][column_idx]
                    .hits
                    .fetch_add(1, Ordering::Relaxed);
                Some(*value)
            }
            None => {
                self.stats[i][column_idx]
                    .misses
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    pub fn set(&self, i: usize, k: usize, key: u64, value: f64) {
        let j = self.top_level_degree - k;
        let cache_cell = &self.cache[i][j];
        let mut cell_map = cache_cell.lock().unwrap();
        cell_map.insert(key, value);
    }

    pub fn clear(&self) {
        for row in &self.cache {
            for cell in row {
                let mut cell_map = cell.lock().unwrap();
                cell_map.clear();
            }
        }
    }

    pub fn stats(&self) -> &Vec<Vec<CacheStats>> {
        &(self.stats)
    }

    pub fn iter(&self) -> BasisCacheIterator {
        BasisCacheIterator {
            cache: &self,
            row: 0,
            col: 0,
            element: 0,
        }
    }
}

pub(crate) struct BasisCacheIterator<'a> {
    cache: &'a BasisCache,
    row: usize,
    col: usize,
    element: usize,
}

impl Iterator for BasisCacheIterator<'_> {
    type Item = ((usize, usize, u64), f64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row >= self.cache.cache.len() {
                return None;
            }
            let row = &self.cache.cache[self.row];
            if self.col >= row.len() {
                self.row += 1;
                self.col = 0;
                continue;
            }
            let cell = &row[self.col];
            let cell_map = cell.lock().unwrap();
            if let Some((key, value)) = cell_map.iter().nth(self.element) {
                let result = (
                    (self.row, self.cache.top_level_degree - self.col, *key),
                    *value,
                );
                self.element += 1;
                return Some(result);
            } else {
                self.col += 1;
                self.element = 0;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cache_iteration() {
        let degree = 3;
        let num_knots = 5;
        let cache = BasisCache::new(num_knots, degree);
        cache.set(0, degree, 5, 6.0);
        cache.set(0, degree - 1, 1, 2.5);
        cache.set(1, degree, 3, 4.0);
        let mut iter = cache.iter();
        assert_eq!(iter.next(), Some(((0, degree, 5), 6.0)));
        assert_eq!(iter.next(), Some(((0, degree - 1, 1), 2.5)));
        assert_eq!(iter.next(), Some(((1, degree, 3), 4.0)));
        assert_eq!(iter.next(), None);
    }
}
