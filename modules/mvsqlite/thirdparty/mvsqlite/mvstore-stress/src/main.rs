mod inmem;
mod tester;

use anyhow::Result;
use backtrace::Backtrace;
use mvclient::{MultiVersionClient, MultiVersionClientConfig};
use clap::Parser;
use tester::{Tester, TesterConfig};
use tracing_subscriber::{fmt::SubscriberBuilder, EnvFilter};

#[derive(Debug, Parser)]
#[clap(name = "mvstore-stress", about = "stress test mvstore")]
struct Opt {
    /// Data plane URL.
    #[clap(long)]
    data_plane: String,

    /// Admin API URL.
    #[clap(long)]
    admin_api: String,

    /// Output log in JSON format.
    #[clap(long)]
    json: bool,

    /// Namespace key.
    #[clap(long, env = "NS_KEY")]
    ns_key: String,

    /// Number of concurrent tasks.
    #[clap(long)]
    concurrency: u64,

    /// Number of iterations.
    #[clap(long)]
    iterations: u64,

    /// Number of pages.
    #[clap(long)]
    pages: u32,

    /// Disable read-your-writes tests.
    #[clap(long)]
    disable_ryw: bool,

    /// Permit HTTP 410 commit responses.
    #[clap(long)]
    permit_410: bool,

    /// Disable read sets.
    #[clap(long)]
    disable_read_set: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let opt = Opt::parse();

    if opt.json {
        SubscriberBuilder::default()
            .with_env_filter(EnvFilter::from_default_env())
            .json()
            .init();
    } else {
        SubscriberBuilder::default()
            .with_env_filter(EnvFilter::from_default_env())
            .pretty()
            .init();
    }

    std::panic::set_hook(Box::new(|info| {
        let bt = Backtrace::new();
        tracing::error!(backtrace = ?bt, info = %info, "panic");
        std::process::abort();
    }));

    let client = MultiVersionClient::new(
        MultiVersionClientConfig {
            data_plane: vec![opt.data_plane.parse()?],
            ns_key: opt.ns_key.clone(),
            ns_key_hashproof: None,
            lock_owner: None,
        },
        reqwest::Client::new(),
    )?;
    let t = Tester::new(
        client.clone(),
        TesterConfig {
            admin_api: opt.admin_api.clone(),
            num_pages: opt.pages,
            disable_ryw: opt.disable_ryw,
            permit_410: opt.permit_410,
            disable_read_set: opt.disable_read_set,
        },
    );
    t.run(opt.concurrency as _, opt.iterations as _).await;
    println!("Test succeeded.");

    // Otherwise we might get "ERROR mvstore_stress: panicked at 'dispatch dropped without returning error', /home/runner/.cargo/registry/src/github.com-1ecc6299db9ec823/hyper-0.14.20/src/client/conn.rs:397:35".
    // https://github.com/losfair/mvsqlite/runs/7676519092
    std::process::exit(0);
}
