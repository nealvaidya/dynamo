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

use crate::mocker::kv_manager::KvManager;
use crate::mocker::protocols::DirectRequest;
use crate::mocker::sequence::ActiveSequence;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{interval, Duration};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

struct SchedulerState {
    waiting_requests: VecDeque<(Uuid, DirectRequest)>,
    running_requests: HashMap<Uuid, ActiveSequence>,
}

/// Manages scheduling of requests using KvManager resources
pub struct Scheduler {
    state: Arc<Mutex<SchedulerState>>,
    kv_manager: Arc<KvManager>, // Store KvManager directly in Scheduler
    active_tokens: Arc<Mutex<HashMap<Uuid, usize>>>,
    token_capacity: usize,
    watermark: f64,
    block_size: usize,
    chunk_size: usize,
    background_handle: Option<JoinHandle<()>>,
    request_tx: mpsc::Sender<DirectRequest>,
    cancellation_token: CancellationToken,
}

impl Scheduler {
    /// Create a new Scheduler with the given parameters
    pub fn new(
        kv_capacity: usize,
        watermark: f64,
        block_size: usize,
        chunk_size: Option<usize>,
        output_tx: Option<mpsc::Sender<Uuid>>,
    ) -> Self {
        // Create KvManager internally
        let kv_manager = KvManager::new(kv_capacity, block_size);

        let token_capacity = 8192;
        let state = Arc::new(Mutex::new(SchedulerState {
            waiting_requests: VecDeque::new(),
            running_requests: HashMap::new(),
        }));

        let kv_manager = Arc::new(kv_manager);
        let chunk_size = chunk_size.unwrap_or(256);

        let active_tokens = Arc::new(Mutex::new(HashMap::new()));

        // Create channel for request handling
        let (request_tx, mut request_rx) = mpsc::channel::<DirectRequest>(1024);

        // Create cancellation token
        let cancellation_token = CancellationToken::new();
        let token_clone = cancellation_token.clone();

        // Create a clone for the background task
        let state_clone = state.clone();
        let kv_manager_clone = kv_manager.clone();
        let watermark_clone = watermark;
        let block_size_clone = block_size;
        let chunk_size_clone = chunk_size;
        let active_tokens_clone = active_tokens.clone();
        let output_tx_clone = output_tx.clone();

        // Spawn background task with cancellation token
        let background_handle = tokio::spawn(async move {
            let mut schedule_interval = interval(Duration::from_millis(5));
            let mut process_interval = interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    Some(request) = request_rx.recv() => {
                        let uuid = Uuid::new_v4();
                        let mut state = state_clone.lock().await;
                        state.waiting_requests.push_back((uuid, request));
                    }

                    _ = process_interval.tick() => {
                        // Acquire locks in order (state first, then active_tokens) to prevent deadlocks
                        let mut state_guard = state_clone.lock().await;
                        let mut active_tokens_guard = active_tokens_clone.lock().await;

                        // Process each running request
                        let mut uuids_to_remove = Vec::new();
                        for (uuid, sequence) in state_guard.running_requests.iter_mut() {
                            // Generate token
                            sequence.generate();

                            // Send UUID notification for each generated token
                            if let Some(tx) = &output_tx_clone {
                                let _ = tx.try_send(*uuid);
                            }

                            // Only set active tokens to 1 if this is the first token generated
                            if sequence.generated_tokens == 1 {
                                active_tokens_guard.insert(*uuid, 1);
                            }

                            // Check if we're done after generating
                            if sequence.generated_tokens >= sequence.max_output_tokens {
                                uuids_to_remove.push(*uuid);
                            }
                        }

                        // Remove completed sequences
                        for uuid in uuids_to_remove {
                            state_guard.running_requests.remove(&uuid);
                            active_tokens_guard.remove(&uuid);
                        }
                    }

                    _ = schedule_interval.tick() => {
                        let mut state_guard = state_clone.lock().await;

                        // Skip if no waiting requests or get the front request
                        let request = match state_guard.waiting_requests.front() {
                            Some((_, request)) => request,
                            None => continue,
                        };

                        let mut active_tokens_guard = active_tokens_clone.lock().await;
                        let current_token_usage: usize = active_tokens_guard.values().sum();

                        // Check scheduling conditions
                        let current_capacity = kv_manager_clone.current_capacity().await;
                        let max_capacity = kv_manager_clone.max_capacity;
                        let input_len = request.tokens.len();

                        let can_schedule =
                            current_token_usage + input_len <= token_capacity &&
                            current_capacity as f64 <= (1.0 - watermark_clone) * max_capacity as f64;

                        if !can_schedule {
                            continue;
                        }

                        // Process the request
                        let (uuid, request) = state_guard.waiting_requests.pop_front().unwrap();
                        let event_tx = kv_manager_clone.get_event_sender();

                        let sequence = ActiveSequence::new(
                            request.tokens,
                            Some(block_size_clone),
                            Some(chunk_size_clone),
                            request.max_output_tokens,
                            Some(event_tx),
                        );

                        active_tokens_guard.insert(uuid, input_len);
                        state_guard.running_requests.insert(uuid, sequence);
                    }

                    // Check for cancellation
                    _ = token_clone.cancelled() => {
                        break;
                    }
                }
            }
        });

        Self {
            state,
            kv_manager,
            active_tokens,
            token_capacity,
            watermark,
            block_size,
            chunk_size,
            background_handle: Some(background_handle),
            request_tx,
            cancellation_token,
        }
    }

    /// Add a new request to the waiting queue
    pub async fn receive_request(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request).await;
    }

    /// Get the count of waiting requests
    pub async fn waiting_count(&self) -> usize {
        let state = self.state.lock().await;
        state.waiting_requests.len()
    }

    /// Get the count of running requests
    pub async fn running_count(&self) -> usize {
        let state = self.state.lock().await;
        state.running_requests.len()
    }
}

// Implement Clone for Scheduler to support sharing between tasks
impl Clone for Scheduler {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            kv_manager: self.kv_manager.clone(),
            active_tokens: self.active_tokens.clone(),
            token_capacity: self.token_capacity,
            watermark: self.watermark,
            block_size: self.block_size,
            chunk_size: self.chunk_size,
            background_handle: None,
            request_tx: self.request_tx.clone(),
            cancellation_token: self.cancellation_token.clone(),
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        // Only the original instance should cancel
        if self.background_handle.is_some() {
            self.cancellation_token.cancel();
        }

        // Abort the background task if this is the original instance
        if let Some(handle) = self.background_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_scheduler_token_generation() {
        std::env::set_var("RUST_LOG", "debug");

        let kv_capacity: usize = 2000;
        let watermark: f64 = 0.01; // 1% watermark
        let block_size: usize = 64;
        let chunk_size: usize = 256;
        let num_requests: usize = 100;
        let input_len: usize = 1000;
        let max_output_tokens: usize = 100;

        // Create channels for token output
        let (output_tx, mut output_rx) = mpsc::channel::<Uuid>(1024);

        // Create scheduler with internal KvManager
        let scheduler = Scheduler::new(
            kv_capacity,
            watermark,
            block_size,
            Some(chunk_size),
            Some(output_tx),
        );

        // Create test requests
        let input_tokens = (0..input_len)
            .map(|_| rand::random::<u32>() % 50000)
            .collect::<Vec<_>>();

        // Submit all requests
        let start_time = std::time::Instant::now();
        for _ in 0..num_requests {
            let request = DirectRequest {
                tokens: input_tokens.clone(),
                max_output_tokens,
            };
            scheduler.receive_request(request).await;
        }

        // Collect all generated tokens (should be num_requests * max_output_tokens)
        let expected_tokens = num_requests * max_output_tokens as usize;
        let mut received_tokens = 0;

        // Wait for all tokens to be generated with a timeout
        let timeout = tokio::time::sleep(Duration::from_secs(30)); // 30 second timeout
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                Some(_) = output_rx.recv() => {
                    received_tokens += 1;
                    if received_tokens == expected_tokens {
                        // Calculate and print elapsed time
                        let elapsed = start_time.elapsed();
                        println!("Test completed in: {:?}", elapsed);
                        break;
                    }
                }
                _ = &mut timeout => {
                    panic!("Test timed out! Received only {} of {} expected tokens", received_tokens, expected_tokens);
                }
            }
        }

        // Verify all tokens were received
        assert_eq!(received_tokens, expected_tokens);

        // Wait to ensure all sequences are cleaned up
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify no running requests remain
        assert_eq!(scheduler.running_count().await, 0);
    }
}
