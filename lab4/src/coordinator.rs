//!
//! coordinator.rs
//! Implementation of 2PC coordinator
//!
extern crate log;
extern crate stderrlog;
extern crate rand;
extern crate ipc_channel;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use coordinator::ipc_channel::ipc::IpcSender as Sender;
use coordinator::ipc_channel::ipc::IpcReceiver as Receiver;
use coordinator::ipc_channel::ipc::TryRecvError;
use coordinator::ipc_channel::ipc::channel;

use message;
use message::MessageType;
use message::ProtocolMessage;
use message::RequestStatus;
use oplog;

/// CoordinatorState
/// States for 2PC state machine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorState {
    Quiescent,
    ReceivedRequest,
    ProposalSent,
    ReceivedVotesAbort,
    ReceivedVotesCommit,
    SentGlobalDecision
}

/// Coordinator
/// Struct maintaining state for coordinator
#[derive(Debug)]
pub struct Coordinator {
    state: CoordinatorState,
    running: Arc<AtomicBool>,
    log: oplog::OpLog,
    id_str: String,
    clients: HashMap<String, Sender<ProtocolMessage>>,
    client_rx: Receiver<ProtocolMessage>,
    num_clients: u64,
    participants: HashMap<String, Sender<ProtocolMessage>>,
    participant_rx: Receiver<ProtocolMessage>,
    num_participants: u64,
    successful_ops: u64,
    failed_ops: u64,
    unknown_ops: u64,
}

///
/// Coordinator
/// Implementation of coordinator functionality
/// Required:
/// 1. new -- Constructor
/// 2. protocol -- Implementation of coordinator side of protocol
/// 3. report_status -- Report of aggregate commit/abort/unknown stats on exit.
/// 4. participant_join -- What to do when a participant joins
/// 5. client_join -- What to do when a client joins
///
impl Coordinator {

    ///
    /// new()
    /// Initialize a new coordinator
    ///
    /// <params>
    ///     log_path: directory for log files --> create a new log there.
    ///     r: atomic bool --> still running?
    ///
    pub fn new(
        log_path: String,
        client_rx: Receiver<ProtocolMessage>,
        participant_rx: Receiver<ProtocolMessage>,
        r: &Arc<AtomicBool>
    ) -> Coordinator {
        Coordinator {
            state: CoordinatorState::Quiescent,
            log: oplog::OpLog::new(log_path),
            running: r.clone(),
            // TODO
            id_str: String::from("coord_0"),
            clients: HashMap::new(),
            client_rx,
            num_clients: 0,
            participants: HashMap::new(),
            participant_rx,
            num_participants: 0,
            successful_ops: 0,
            failed_ops: 0,
            unknown_ops: 0,
        }
    }

    ///
    /// participant_join()
    /// Adds a new participant for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    pub fn participant_join(&mut self, name: &String, tx: Sender<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);

        self.participants.insert(name.to_string(), tx);
        self.num_participants += 1;
    }

    ///
    /// client_join()
    /// Adds a new client for the coordinator to keep track of
    ///
    /// HINT: Keep track of any channels involved!
    /// HINT: You may need to change the signature of this function
    ///
    //pub fn client_join(&mut self, name: &String) {
    pub fn client_join(&mut self, name: &String, tx: Sender<ProtocolMessage>) {
        assert!(self.state == CoordinatorState::Quiescent);

        self.clients.insert(name.to_string(), tx);
        self.num_clients += 1;
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this coordinator before exiting.
    ///
    pub fn report_status(&mut self) {
        // TODO: Collect actual stats
        let successful_ops: u64 = self.successful_ops;
        let failed_ops: u64 = self.failed_ops;
        let unknown_ops: u64 = self.unknown_ops;

        println!("coordinator     :\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", successful_ops, failed_ops, unknown_ops);
    }

    ///
    /// protocol()
    /// Implements the coordinator side of the 2PC protocol
    /// HINT: If the simulation ends early, don't keep handling requests!
    /// HINT: Wait for some kind of exit signal before returning from the protocol!
    ///
    pub fn protocol(&mut self) {
        while self.num_clients > 0 && self.running.load(Ordering::SeqCst) {
            let Ok(request) = self.client_rx.try_recv() else {
                continue;
            };

            info!("{}::Received request from {}", self.id_str.clone(), request.senderid);

            match request.mtype {
                MessageType::ClientRequest => {
                    let status = self.make_request(&request);
                    self.notify_participants(&request, status);
                    self.send_client_status(&request, status);
                }
                MessageType::CoordinatorExit => {
                    info!("{}::Received shutdown from {}", self.id_str.clone(), request.senderid.clone());
                    self.num_clients -= 1;
                    //self.clients.remove(&request.senderid);
                }
                _ => println!("HANDLE ME!")
            }
        }

        // Shutdown participants.
        for (_, participant_tx) in &self.participants {
            self.send_shutdown(participant_tx);
        }

        // Shutdown any remaining clients.
        for (_, client_tx) in &self.clients {
            self.send_shutdown(client_tx);
        }

        self.report_status();
    }

    pub fn make_request(&mut self, request: &message::ProtocolMessage) -> RequestStatus {
        let mut received = 0;
        let mut abort = false;

        for (participant, tx) in &self.participants {
            let pm = ProtocolMessage::generate(
                MessageType::CoordinatorPropose,
                request.txid.clone(),
                self.id_str.clone(),
                request.opid,
            );
            info!("{}::Sending propose to {}", self.id_str.clone(), participant);
            self.log.append(MessageType::CoordinatorPropose, request.txid.clone(), self.id_str.clone(), request.opid);
            tx.send(pm).unwrap();
        }

        let now = Instant::now();

        loop {
            if let Ok(result) = self.participant_rx.try_recv() {
                // Ensure result is for our requested operation and not a straggler from
                // a previous one.
                if result.txid != request.txid {
                    continue;
                }

                info!("{}::Received {:?} from {}", self.id_str.clone(), result.mtype, result.senderid.clone());
                if result.mtype == MessageType::ParticipantVoteAbort {
                    abort = true;
                }

                received += 1;
                if received == self.num_participants {
                    break;
                }
            } else if Instant::now() - now >= Duration::from_millis(2) {
                info!("{}::Participants timed out", self.id_str.clone());
                abort = true;
                break;
            }
        }

        if abort {
            self.failed_ops += 1;
            RequestStatus::Aborted
        } else {
            self.successful_ops += 1;
            RequestStatus::Committed
        }
    }

    pub fn send_client_status(&mut self, request: &ProtocolMessage, status: RequestStatus) {
        let client_tx = self.clients.get(&request.senderid).unwrap();

        let mtype = match status {
            RequestStatus::Committed => MessageType::ClientResultCommit,
            _ => MessageType::ClientResultAbort,
        };
        let pm = ProtocolMessage::generate(mtype, request.txid.clone(), self.id_str.clone(), request.opid);

        self.log.append(mtype, request.txid.clone(), self.id_str.clone(), request.opid);
        client_tx.send(pm).unwrap();
    }

    pub fn notify_participants(&mut self, request: &ProtocolMessage, status: RequestStatus) {

        let mtype = match status {
            RequestStatus::Committed => MessageType::CoordinatorCommit,
            _ => MessageType::CoordinatorAbort,
        };

        self.log.append(mtype, request.txid.clone(), self.id_str.clone(), request.opid);

        for (_, participant_tx) in &self.participants {
            let pm = ProtocolMessage::generate(mtype, request.txid.clone(), self.id_str.clone(), request.opid);
            participant_tx.send(pm).unwrap();
        }
    }

    pub fn send_shutdown(&self, tx: &Sender<ProtocolMessage>) {
        let pm = ProtocolMessage::generate(
            MessageType::CoordinatorExit,
            String::from(""),
            self.id_str.clone(),
            0
        );
        tx.send(pm).ok();
    }
}
