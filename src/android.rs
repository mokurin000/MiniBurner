//! Android entry point for the QR code generator app.

use winio::prelude::*;

use crate::model::MainModel;

/// Native entry point called by the Android system.
#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    tracing_subscriber::registry()
        .with(tracing_android_trace::AndroidTraceLayer::new())
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_max_level(LevelFilter::INFO),
        )
        .try_init()
        .ok();

    let app = App::builder()
        .android_app(app)
        .build()
        .expect("cannot create app");
    app.spawn(|| async {
        _ = MainModel::run_until_event(()).await;
    })
}
