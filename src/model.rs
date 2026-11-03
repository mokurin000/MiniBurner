//! Main GUI component for the QR code generator.
use std::random::{Rng as _, SystemRng};

use sha2::Digest;
use winio::prelude::*;

use crate::Result;

/// Root component of the application UI.
pub struct MainModel {
    /// The main application window.
    window: Child<Window>,
    /// label
    label: Child<Label>,
}

pub enum MainMessage {
    /// Nothing to do
    Noop,
    /// Main window has been resized
    Resize,
    /// Theme changed
    ThemeChanged,
    /// Close main window
    Close,
}

impl Component for MainModel {
    type Error = color_eyre::Report;
    type Event = ();
    type Init<'a> = ();
    type Message = MainMessage;

    async fn init(_init: Self::Init<'_>, _sender: &ComponentSender<Self>) -> Result<Self> {
        // Note: color-eyre does not enable VT100 on Windows on its own
        color_eyre::install()?;

        init! {
            window: Window = (()) => {
                text: "MiniBurner",
                size: Size::new(300.0, 100.0),

                #[cfg(all(windows, feature = "winui"))]
                backdrop: Backdrop::Mica,
            },
            label: Label = (&window) => {
                text: "Burning CPU...",
            },
        }

        window.show()?;

        let _ = std::iter::once(|| {
            #[repr(align(128))]
            struct Aligned<T>(T);

            // 256 KiB per thread
            let mut buf = Aligned::<_>([0u8; 256 * 1024]);
            SystemRng.fill_bytes(&mut buf.0);

            loop {
                // AVX2 has wider support than SHA-NI on the desktop CPU
                #[cfg(target_arch = "x86_64")]
                let mut digest = sha2::Sha512::new();
                // SHA512 requires ARMv8.4-A or later (2019~2021)
                // SHA256 requires ARMv8.0-A or later (2013~2014)
                #[cfg(not(target_arch = "x86_64"))]
                let mut digest = sha2::Sha256::new();

                std::hint::black_box(digest.update(buf.0));
                digest.finalize();
            }
        })
        .cycle()
        .take(num_cpus::get())
        .map(std::thread::spawn)
        .collect::<Vec<_>>();

        Ok(Self { window, label })
    }

    async fn start(&mut self, sender: &ComponentSender<Self>) -> ! {
        // listen to events
        start! {
            sender, default: MainMessage::Noop,
            self.window => {
                WindowEvent::Resize => MainMessage::Resize,
                WindowEvent::Close => MainMessage::Close,
                WindowEvent::ThemeChanged => MainMessage::ThemeChanged,
            },
        }
    }

    async fn update_children(&mut self) -> Result<bool> {
        // update the window and functional children
        update_children!(self.window)
    }

    async fn update(
        &mut self,
        message: Self::Message,
        sender: &ComponentSender<Self>,
    ) -> Result<bool> {
        // deal with custom messages
        match message {
            MainMessage::Noop => Ok(false),
            MainMessage::ThemeChanged => Ok(false),
            MainMessage::Resize => Ok(true),
            MainMessage::Close => {
                // the root component output stops the application
                sender.output(());
                // need not to call `render`
                Ok(false)
            }
        }
    }

    fn render(&mut self, _sender: &ComponentSender<Self>) -> Result<()> {
        let csize = self.window.client_size()?;

        let mut layout = layout! {
            Grid::new(
                vec![GridLength::Stretch(1.0)],
                vec![GridLength::Stretch(1.0)],
            ),
            self.label => {
                halign: HAlign::Center,
                valign: VAlign::Center,
            },
        };
        layout.set_size(csize)?;
        Ok(())
    }

    fn render_children(&mut self) -> Result<()> {
        Ok(self.window.render()?)
    }
}
