//! Main GUI component for the QR code generator.
use std::random::{Rng as _, SystemRng};
use std::time::Duration;

use sha2::Digest;
use winio::prelude::*;

use crate::Result;
use crate::donut::DonutRenderer;

/// How far the donut spins around the screen's Z axis on every tick.
const SPIN_DEGREES_PER_TICK: f32 = 6.0;
/// Interval between two rotation ticks.
const TICK_INTERVAL: Duration = Duration::from_millis(10);

/// Selectable fur quality tiers: label and number of fur shells.
const FUR_QUALITY: [(&str, u32); 4] = [
    ("Low End", 128),
    ("Medium", 256),
    ("High End", 768),
    ("Extreme", 1024),
];
/// Quality tier selected when the window opens.
const DEFAULT_FUR_QUALITY: usize = 2;

/// Root component of the application UI.
pub struct MainModel {
    /// The main application window.
    window: Child<Window>,
    canvas: Child<WgpuCanvas>,
    /// Fur quality selector, shown until the burn starts.
    quality: Child<ComboBox>,
    /// Starts the CPU and GPU burn.
    start_button: Child<Button>,
    /// Skip the CPU burn
    skip_cpu_burn: Child<CheckBox>,

    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Created when the burn starts; it also gates all GPU rendering.
    renderer: Option<DonutRenderer>,
    /// Whether the CPU and GPU burn is running.
    burning: bool,
    /// Rotation of the donut around the screen's Z axis, in radians.
    spin: f32,

    clockwise: bool,
}

pub enum MainMessage {
    /// Start the CPU and GPU burn with the selected fur quality
    Start,
    /// Advance the donut rotation by one tick
    Tick,
    /// Main window has been resized
    Resize,
    /// Theme changed
    ThemeChanged,
    /// Close main window
    Close,
    /// Inverse rotate direction
    InverseDirection,
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

        let monitor = Monitor::all()?;
        let size = monitor
            .first()
            .map(|m| m.region().size)
            .unwrap_or(Size::new(1920., 1080.));
        let quality_labels = FUR_QUALITY.map(|(label, _)| label);

        init! {
            window: Window = (()) => {
                text: "MiniBurner",
                size: size,

                #[cfg(all(windows, feature = "winui"))]
                backdrop: Backdrop::Mica,
            },
            skip_cpu_burn: CheckBox = (&window) => {
                text: "Skip CPU Burn",
            },
            canvas: WgpuCanvas = (&window),
            quality: ComboBox = (&window) => {
                items: quality_labels,
                selection: DEFAULT_FUR_QUALITY,
            },
            start_button: Button = (&window) => {
                text: "Start",
            },
        }

        canvas.bind(&instance, &adapter);
        canvas.set_size(window.client_size()?)?;
        // The canvas only comes up once the burn starts.
        canvas.set_visible(false)?;
        window.show()?;

        Ok(Self {
            window,
            canvas,
            quality,
            start_button,
            skip_cpu_burn,
            device,
            queue,
            renderer: None,
            burning: false,
            clockwise: true,
            spin: 0.0,
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
            self.canvas => {
                WgpuCanvasEvent::MouseDown(_) => MainMessage::InverseDirection,
            },
            self.quality => {},
            self.skip_cpu_burn => {},
            self.start_button => {
                ButtonEvent::Click => MainMessage::Start,
            }
        }
    }

    async fn update_children(&mut self) -> Result<bool> {
        // update the window and functional children
        update_children!(
            self.window,
            self.canvas,
            self.quality,
            self.start_button,
            self.skip_cpu_burn,
        )
    }

    async fn update(
        &mut self,
        message: Self::Message,
        sender: &ComponentSender<Self>,
    ) -> Result<bool> {
        // deal with custom messages
        match message {
            MainMessage::Start => {
                if self.burning {
                    return Ok(false);
                }

                let selected = self.quality.selection()?.unwrap_or(DEFAULT_FUR_QUALITY);
                let fur_layers = FUR_QUALITY
                    .get(selected)
                    .map(|(_, layers)| *layers)
                    .unwrap_or(FUR_QUALITY[DEFAULT_FUR_QUALITY].1);

                // Build the renderer first: if it fails, the UI stays usable.
                self.renderer = Some(DonutRenderer::new(&self.device, fur_layers)?);
                self.burning = true;

                self.quality.set_visible(false)?;
                self.start_button.set_visible(false)?;
                self.canvas.set_visible(true)?;

                if !self.skip_cpu_burn.is_checked()? {
                    spawn_cpu_burners();
                }
                spawn_rotation_ticker(sender.clone());
                Ok(true)
            }
            MainMessage::Tick => {
                if !self.burning {
                    return Ok(false);
                }

                // counterclockwise
                if !self.clockwise {
                    self.spin = (self.spin + SPIN_DEGREES_PER_TICK.to_radians())
                        .rem_euclid(std::f32::consts::TAU);
                // clockwise
                } else {
                    self.spin = (self.spin - SPIN_DEGREES_PER_TICK.to_radians())
                        .rem_euclid(std::f32::consts::TAU);
                }

                Ok(true)
            }
            MainMessage::InverseDirection => {
                self.clockwise = !self.clockwise;
                Ok(false)
            }
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
        if csize.width <= 0.0 || csize.height <= 0.0 {
            return Ok(());
        }
        if self.canvas.size()? != csize {
            self.canvas.set_size(csize)?;
        }

        if !self.burning {
            let mut inner_layout = layout! {
                StackPanel::new(Orient::Vertical),
                self.quality => {
                    margin: Margin::new_all_same(2.0),
                },
                self.skip_cpu_burn => {
                    margin: Margin::new_all_same(2.0),
                },
                self.start_button => {
                    margin: Margin::new_all_same(2.0),
                },
            };
            let mut layout = layout! {
                Grid::new(vec![GridLength::Stretch(1.0)], vec![GridLength::Stretch(1.0)]),
                inner_layout => {
                    valign: VAlign::Center,
                    halign: HAlign::Center,
                },
            };
            layout.set_size(csize)?;
            return Ok(());
        }

        // Render the donut into the canvas' swapchain texture, then hand the
        // frame over to the widget.
        let Some(frame) = self.canvas.acquire(&self.device) else {
            return Ok(());
        };
        let size = (frame.texture.width(), frame.texture.height());
        self.renderer
            .as_mut()
            .expect("the renderer exists once the burn starts")
            .render(
                &self.device,
                &self.queue,
                &frame.view,
                size,
                frame.format,
                self.spin,
            );
        self.canvas.present(&self.queue)?;
        Ok(())
    }

    fn render_children(&mut self) -> Result<()> {
        self.window.render()?;
        if !self.burning {
            self.quality.render()?;
            self.skip_cpu_burn.render()?;
            self.start_button.render()?;
        }
        Ok(())
    }
}

/// Start one SHA burner thread per CPU core.
///
/// The thread handles are dropped on purpose: the burners detach and keep
/// running for the rest of the process' lifetime.
fn spawn_cpu_burners() {
    for _ in 0..num_cpus::get() {
        let _ = std::thread::spawn(|| {
            #[repr(align(128))]
            struct Aligned<T>(T);

            // 256 KiB per thread
            let mut buf = Aligned::<[u8; 256 * 1024]>([0u8; 256 * 1024]);
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
        });
    }
}

/// Post a [`MainMessage::Tick`] every [`TICK_INTERVAL`] on the compio runtime.
fn spawn_rotation_ticker(sender: ComponentSender<MainModel>) {
    compio::runtime::spawn(async move {
        loop {
            compio::time::sleep(TICK_INTERVAL).await;
            sender.post(MainMessage::Tick);
        }
    })
    .detach();
}
