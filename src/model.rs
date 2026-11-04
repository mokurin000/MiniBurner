//! Main GUI component for the QR code generator.
use std::random::{Rng as _, SystemRng};

use sha2::Digest;
use winio::prelude::*;

use crate::Result;
use crate::donut::DonutRenderer;

/// Root component of the application UI.
pub struct MainModel {
    /// The main application window.
    window: Child<Window>,
    canvas: Child<WgpuCanvas>,

    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: DonutRenderer,
}

pub enum MainMessage {
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

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        let renderer = DonutRenderer::new(&device)?;

        init! {
            window: Window = (()) => {
                text: "MiniBurner",
                size: Size::new(800.0, 600.0),

                #[cfg(all(windows, feature = "winui"))]
                backdrop: Backdrop::Mica,
            },
            canvas: WgpuCanvas = (&window),
        }
        canvas.bind(&instance, &adapter);
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

        Ok(Self {
            window,
            canvas,
            device,
            queue,
            renderer,
        })
    }

    async fn start(&mut self, sender: &ComponentSender<Self>) -> ! {
        // listen to events
        start! {
            sender,
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
        if self.canvas.size()? != csize {
            self.canvas.set_size(csize)?;
        }
        if csize.width <= 0.0 || csize.height <= 0.0 {
            return Ok(());
        }

        // Render the donut into the canvas' swapchain texture, then hand the
        // frame over to the widget.
        let Some(frame) = self.canvas.acquire(&self.device) else {
            return Ok(());
        };
        let size = (frame.texture.width(), frame.texture.height());
        self.renderer
            .render(&self.device, &self.queue, &frame.view, size, frame.format);
        self.canvas.present(&self.queue)?;
        Ok(())
    }

    fn render_children(&mut self) -> Result<()> {
        Ok(self.window.render()?)
    }
}
