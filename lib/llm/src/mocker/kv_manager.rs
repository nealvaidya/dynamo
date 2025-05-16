// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::mocker::evictor::LRUEvictor;
use crate::mocker::protocols::{MoveBlock, UniqueBlock};
use std::collections::{HashMap, HashSet};
use std::panic;
use std::time::Instant;

/// Mock implementation of worker for testing and simulation
pub struct KvManager {
    pub max_capacity: usize,
    pub block_size: usize,
    pub active_blocks: HashMap<UniqueBlock, usize>,
    pub inactive_blocks: LRUEvictor<UniqueBlock>,
    pub start_time: Instant,
    pub all_blocks: HashSet<UniqueBlock>,
}

impl KvManager {
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        let active_blocks = HashMap::new();
        let inactive_blocks = LRUEvictor::new();
        let start_time = Instant::now();
        let all_blocks = HashSet::new();

        KvManager {
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            start_time,
            all_blocks,
        }
    }

    /// Process a MoveBlock instruction synchronously
    pub fn process(&mut self, event: MoveBlock) {
        match event {
            MoveBlock::Use(hashes, _) => {
                for hash in hashes {
                    // First check if it already exists in active blocks
                    if let Some(ref_count) = self.active_blocks.get_mut(&hash) {
                        // Block already active, just increment reference count
                        *ref_count += 1;
                        continue;
                    }

                    // Then check if it exists in inactive and move it to active if found
                    if self.inactive_blocks.remove(&hash) {
                        // Insert into active with reference count 1
                        self.active_blocks.insert(hash.clone(), 1);
                        continue;
                    }

                    // Get counts for capacity check
                    let active_count = self.active_blocks.len();
                    let inactive_count = self.inactive_blocks.num_objects();

                    // If at max capacity, evict the oldest entry from inactive blocks
                    if active_count + inactive_count >= self.max_capacity {
                        if let Some(evicted) = self.inactive_blocks.evict() {
                            // Remove evicted block from all_blocks
                            self.all_blocks.remove(&evicted);
                        } else {
                            panic!("max capacity reached and no free blocks left");
                        }
                    }

                    // Now insert the new block in active blocks with reference count 1
                    self.active_blocks.insert(hash.clone(), 1);
                    // Add to all_blocks as it's a new block
                    self.all_blocks.insert(hash);
                }
            }
            MoveBlock::Destroy(hashes) => {
                // Loop in inverse direction
                for hash in hashes.into_iter().rev() {
                    self.active_blocks.remove(&hash);
                    // Remove from all_blocks when destroyed
                    self.all_blocks.remove(&hash);
                }
            }
            MoveBlock::Deref(hashes) => {
                // Loop in inverse direction
                for hash in hashes.into_iter().rev() {
                    // Decrement reference count and check if we need to move to inactive
                    if let Some(ref_count) = self.active_blocks.get_mut(&hash) {
                        *ref_count -= 1;

                        // If reference count reaches zero, remove from active and move to inactive
                        if *ref_count == 0 {
                            self.active_blocks.remove(&hash);
                            self.inactive_blocks
                                .insert(hash, self.start_time.elapsed().as_secs_f64());
                        }
                    }
                }
            }
            MoveBlock::Promote(uuid, hash) => {
                let uuid_block = UniqueBlock::PartialBlock(uuid);
                let hash_block = UniqueBlock::FullBlock(hash);

                // Check if the UUID block exists in active blocks
                if let Some(ref_count) = self.active_blocks.remove(&uuid_block) {
                    // Replace with hash block, keeping the same reference count
                    self.active_blocks.insert(hash_block.clone(), ref_count);

                    // Update all_blocks
                    self.all_blocks.remove(&uuid_block);
                    self.all_blocks.insert(hash_block);
                }
            }
        }
    }

    /// Get the count of blocks in the input list that aren't in all_blocks
    pub fn probe_new_blocks(&self, blocks: Vec<UniqueBlock>) -> usize {
        blocks.into_iter().filter(|block| !self.all_blocks.contains(block)).count()
    }

    /// Get the current capacity (active blocks + inactive blocks)
    pub fn current_capacity(&self) -> usize {
        let active = self.active_blocks.len();
        let inactive = self.inactive_blocks.num_objects();
        active + inactive
    }

    /// Get the keys of inactive blocks
    pub fn get_inactive_blocks(&self) -> Vec<UniqueBlock> {
        self.inactive_blocks.free_table.keys().cloned().collect()
    }

    /// Get the keys of active blocks
    pub fn get_active_blocks(&self) -> Vec<UniqueBlock> {
        self.active_blocks.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panic_on_max_capacity() {
        // Create a KvManager with 10 blocks capacity
        let mut manager = KvManager::new(10, 16);

        // Helper function to use multiple blocks
        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(MoveBlock::Use(blocks, None));
        }

        // First use 10 blocks (0 to 9) in a batch
        use_blocks(&mut manager, (0..10).collect());

        // Verify we are at capacity
        assert_eq!(manager.current_capacity(), 10);

        // The 11th block should cause a panic
        let result = panic::catch_unwind(move || {
            use_blocks(&mut manager, vec![10]);
        });

        // Verify that a panic occurred
        assert!(
            result.is_err(),
            "Expected a panic when exceeding max capacity"
        );
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        // Create a KvManager with 10 blocks capacity
        let mut manager = KvManager::new(10, 16);

        // Helper function to use multiple blocks
        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(MoveBlock::Use(blocks, None));
        }

        // Helper function to destroy multiple blocks
        fn destroy_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(MoveBlock::Destroy(blocks));
        }

        // Helper function to deref multiple blocks
        fn deref_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(MoveBlock::Deref(blocks));
        }

        // Helper function to check if active blocks contain expected blocks with expected ref counts
        fn assert_active_blocks(manager: &KvManager, expected_blocks: &[(u64, usize)]) {
            assert_eq!(
                manager.active_blocks.len(),
                expected_blocks.len(),
                "Active blocks count doesn't match expected"
            );

            for &(id, ref_count) in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    manager.active_blocks.contains_key(&block),
                    "Block {} not found in active blocks",
                    id
                );
                assert_eq!(
                    manager.active_blocks.get(&block),
                    Some(&ref_count),
                    "Block {} has wrong reference count",
                    id
                );
            }
        }

        // Helper function to check if inactive blocks contain expected blocks
        fn assert_inactive_blocks(
            manager: &KvManager,
            expected_size: usize,
            expected_blocks: &[u64],
        ) {
            let inactive_blocks = manager.get_inactive_blocks();
            let inactive_blocks_count = manager.inactive_blocks.num_objects();

            assert_eq!(
                inactive_blocks_count, expected_size,
                "Inactive blocks count doesn't match expected"
            );

            for &id in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    inactive_blocks.contains(&block),
                    "Block {} not found in inactive blocks",
                    id
                );
            }
        }

        // First use blocks 0, 1, 2, 3, 4 in a batch
        use_blocks(&mut manager, (0..5).collect());

        // Then use blocks 0, 1, 5, 6 in a batch
        use_blocks(&mut manager, vec![0, 1, 5, 6]);

        // Check that the blocks 0 and 1 are in active blocks, both with reference counts of 2
        assert_active_blocks(
            &manager,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        // Now destroy block 4
        destroy_blocks(&mut manager, vec![4]);

        // And deref blocks 3, 2, 1, 0 in this order as a batch
        deref_blocks(&mut manager, vec![0, 1, 2, 3]);

        // Check that the inactive_blocks is size 2 (via num_objects) and contains 3 and 2
        assert_inactive_blocks(&manager, 2, &[3, 2]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (5, 1), (6, 1)]);

        // Now destroy block 6
        destroy_blocks(&mut manager, vec![6]);

        // And deref blocks 5, 1, 0 as a batch
        deref_blocks(&mut manager, vec![0, 1, 5]);

        // Check that the inactive_blocks is size 5, and contains 0, 1, 2, 3, 5
        assert_inactive_blocks(&manager, 5, &[0, 1, 2, 3, 5]);
        assert_active_blocks(&manager, &[]);

        // Now use 0, 1, 2, 7, 8, 9 as a batch
        use_blocks(&mut manager, vec![0, 1, 2, 7, 8, 9]);

        // Check that the inactive_blocks is size 2, and contains 3 and 5
        assert_inactive_blocks(&manager, 2, &[3, 5]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);

        // Test the new_blocks method - only block 4 should be new out of [0,1,2,3,4]
        let blocks_to_check: Vec<UniqueBlock> = vec![0, 1, 2, 3, 4].into_iter().map(UniqueBlock::FullBlock).collect();
        assert_eq!(manager.probe_new_blocks(blocks_to_check), 1);

        // Now use blocks 10, 11, 12 as a batch
        use_blocks(&mut manager, vec![10, 11, 12]);

        // Check that the inactive_blocks is size 1 and contains only 5
        assert_inactive_blocks(&manager, 1, &[5]);
    }
}
