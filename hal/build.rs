fn main() {
    // Only compile the VL53L8CX C library when the feature is active.
    if std::env::var("CARGO_FEATURE_VL53L8CX").is_ok() {
        let uld = "src/vl53l8cx_uld";

        cc::Build::new()
            .file(format!("{uld}/vl53l8cx_api.c"))
            .file(format!("{uld}/platform.c"))
            .file(format!("{uld}/wrapper.c"))
            .include(uld)
            // Suppress warnings from ST's code we don't own.
            .flag_if_supported("-Wno-unused-parameter")
            .flag_if_supported("-Wno-unused-variable")
            .flag_if_supported("-Wno-missing-field-initializers")
            .compile("vl53l8cx");

        // Re-run build script if C sources change.
        println!("cargo:rerun-if-changed={uld}/vl53l8cx_api.c");
        println!("cargo:rerun-if-changed={uld}/vl53l8cx_api.h");
        println!("cargo:rerun-if-changed={uld}/vl53l8cx_buffers.h");
        println!("cargo:rerun-if-changed={uld}/platform.c");
        println!("cargo:rerun-if-changed={uld}/platform.h");
        println!("cargo:rerun-if-changed={uld}/wrapper.c");
        println!("cargo:rerun-if-changed={uld}/wrapper.h");
    }
}
