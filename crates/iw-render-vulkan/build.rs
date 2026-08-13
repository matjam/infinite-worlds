//! Compiles the GLSL in `<workspace>/shaders` to SPIR-V in `OUT_DIR` with `glslc`.

use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: [&str; 4] = ["globe.vert", "globe.frag", "star.vert", "star.frag"];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest
        .parent()
        .and_then(Path::parent)
        .expect("crate lives at <workspace>/crates/<name>")
        .join("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    for name in SHADERS {
        let src = shader_dir.join(name);
        println!("cargo:rerun-if-changed={}", src.display());
        let dst = out_dir.join(format!("{name}.spv"));
        let out = Command::new("glslc")
            .arg("--target-env=vulkan1.2")
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .unwrap_or_else(|e| panic!("failed to run glslc (is it on PATH?): {e}"));
        if !out.status.success() {
            panic!(
                "glslc failed for {}:\n{}",
                src.display(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
