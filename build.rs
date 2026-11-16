fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        let mut res = winresource::WindowsResource::new();
        res.set_language(1033); // en-US
        res.set_icon("res/logo.ico");
        res.compile().unwrap();
    }
}
