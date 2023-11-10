#[macro_use]
extern crate log;
extern crate stderrlog;
extern crate clap;
extern crate ctrlc;
extern crate ipc_channel;
use std::env;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::process::{Child,Command};
use ipc_channel::ipc::IpcSender as Sender;
use ipc_channel::ipc::IpcReceiver as Receiver;
use ipc_channel::ipc::IpcOneShotServer;
use ipc_channel::ipc::channel;
pub mod message;
pub mod oplog;
pub mod coordinator;
pub mod participant;
pub mod client;
pub mod checker;
pub mod tpcoptions;
use message::ProtocolMessage;

///
/// pub fn spawn_child_and_connect(child_opts: &mut tpcoptions::TPCOptions) -> (std::process::Child, Sender<ProtocolMessage>, Receiver<ProtocolMessage>)
///
///     child_opts: CLI options for child process
///
/// 1. Set up IPC
/// 2. Spawn a child process using the child CLI options
/// 3. Do any required communication to set up the parent / child communication channels
/// 4. Return the child process handle and the communication channels for the parent
///
/// HINT: You can change the signature of the function if necessary
///
fn spawn_child_and_connect(
    child_opts: &mut tpcoptions::TPCOptions, coord_tx: &Sender<ProtocolMessage>
) -> (Child, Sender<ProtocolMessage>) {
    let (server, server_name) = IpcOneShotServer::<Sender<(Sender<ProtocolMessage>, Receiver<ProtocolMessage>)>>::new().unwrap();
    let (tx, rx) = channel().unwrap();

    child_opts.ipc_path = server_name;

    let child = Command::new(env::current_exe().unwrap())
        .args(child_opts.as_vec())
        .spawn()
        .expect("Failed to execute child process");

    let (_, tx_rx_sender) = server.accept().unwrap();
    tx_rx_sender.send((coord_tx.clone(), rx)).unwrap();

    (child, tx)
}

///
/// pub fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>)
///
///     opts: CLI options for this process
///
/// 1. Connect to the parent via IPC
/// 2. Do any required communication to set up the parent / child communication channels
/// 3. Return the communication channels for the child
///
/// HINT: You can change the signature of the function if necessasry
///
fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
    let socket = opts.ipc_path.clone();
    let sender = Sender::connect(socket).unwrap();
    let (tx_rx_sender, tx_rx_receiver) = channel().unwrap();

    sender.send(tx_rx_sender).unwrap();
    let (coord_tx, child_rx) = tx_rx_receiver.recv().unwrap();

    (coord_tx, child_rx)
}

///
/// pub fn run(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Creates a new coordinator
/// 2. Spawns and connects to new clients processes and then registers them with
///    the coordinator
/// 3. Spawns and connects to new participant processes and then registers them
///    with the coordinator
/// 4. Starts the coordinator protocol
/// 5. Wait until the children finish execution
///
fn run(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    let coord_log_path = format!("{}//{}", opts.log_path, "coordinator.log");

    let (coord_client_tx, coord_client_rx) = channel().unwrap();
    let (coord_participant_tx, coord_participant_rx) = channel().unwrap();
    let mut coord = coordinator::Coordinator::new(coord_log_path, coord_client_rx, coord_participant_rx, &running);

    for id in 0..opts.num_participants {
        let mut child_opts = opts.clone();
        child_opts.mode = String::from("participant");
        child_opts.num = id;

        let (_, child_tx) = spawn_child_and_connect(&mut child_opts, &coord_participant_tx);

        let participant_id = format!("participant_{}", id);
        coord.participant_join(&participant_id, child_tx);
    }

    for id in 0..opts.num_clients {
        let mut child_opts = opts.clone();
        child_opts.mode = String::from("client");
        child_opts.num = id;

        let (_, child_tx) = spawn_child_and_connect(&mut child_opts, &coord_client_tx);

        let client_id = format!("client_{}", id);
        coord.client_join(&client_id, child_tx);
    }

    coord.protocol();
}

///
/// pub fn run_client(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Connects to the coordinator to get tx/rx
/// 2. Constructs a new client
/// 3. Starts the client protocol
///
fn run_client(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    let (coord_tx, client_rx) = connect_to_coordinator(opts);

    let id_str = format!("client_{}", opts.num);
    let mut client = client::Client::new(id_str, coord_tx, client_rx, running);
    client.protocol(opts.num_requests);
}

///
/// pub fn run_participant(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Connects to the coordinator to get tx/rx
/// 2. Constructs a new participant
/// 3. Starts the participant protocol
///
fn run_participant(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    let participant_id_str = format!("participant_{}", opts.num);
    let participant_log_path = format!("{}//{}.log", opts.log_path, participant_id_str);

    let (coord_tx, participant_rx) = connect_to_coordinator(opts);
    let mut participant = participant::Participant::new(
        participant_id_str,
        participant_log_path,
        running,
        opts.send_success_probability,
        opts.operation_success_probability,
        coord_tx,
        participant_rx,
    );

    participant.protocol();
}

fn main() {
    // Parse CLI arguments
    let opts = tpcoptions::TPCOptions::new();
    // Set-up logging and create OpLog path if necessary
    stderrlog::new()
            .module(module_path!())
            .quiet(false)
            .timestamp(stderrlog::Timestamp::Millisecond)
            .verbosity(opts.verbosity)
            .init()
            .unwrap();
    match fs::create_dir_all(opts.log_path.clone()) {
        Err(e) => error!("Failed to create log_path: \"{:?}\". Error \"{:?}\"", opts.log_path, e),
        _ => (),
    }

    // Set-up Ctrl-C / SIGINT handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    let m = opts.mode.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        if m == "run" {
            print!("\n");
        }
    }).expect("Error setting signal handler!");

    // Execute main logic
    match opts.mode.as_ref() {
        "run" => run(&opts, running),
        "client" => run_client(&opts, running),
        "participant" => run_participant(&opts, running),
        "check" => checker::check_last_run(opts.num_clients, opts.num_requests, opts.num_participants, &opts.log_path),
        _ => panic!("Unknown mode"),
    }
}
