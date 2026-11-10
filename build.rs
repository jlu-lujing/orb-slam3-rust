// Build script.
//
// Currently just unpacks the ORB vocabulary if needed. The vocabulary is
// shipped compressed (`vocabulary/orbvoc.txt.tar.gz`, ~40 MiB) and consumed
// at runtime as plain text (`vocabulary/orbvoc.txt`, ~140 MiB). We unpack
// once and re-unpack only when the archive is newer than the extracted file.

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // The `opencv` crate generates its bindings from the installed OpenCV, so
    // the arity of some calls depends on that version. OpenCV 4.9 added the
    // trailing `AlgorithmHint` argument to `gaussianBlur`; expose it as a cfg
    // so the source can build against both the 4.11 used on the development
    // host and the 4.8 shipped on the Jetson target.
    println!("cargo:rerun-if-env-changed=OPENCV_PKGCONFIG_NAME");
    let opencv_name =
        std::env::var("OPENCV_PKGCONFIG_NAME").unwrap_or_else(|_| "opencv4".to_string());
    let version = std::process::Command::new("pkg-config")
        .args(["--modversion", &opencv_name])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok());
    let ge_49 = version
        .as_deref()
        .and_then(parse_major_minor)
        .is_some_and(|(major, minor)| major > 4 || (major == 4 && minor >= 9));
    println!("cargo:rustc-check-cfg=cfg(opencv_ge_49)");
    if ge_49 {
        println!("cargo:rustc-cfg=opencv_ge_49");
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let voc_dir = manifest_dir.join("vocabulary");
    let archive = voc_dir.join("orbvoc.txt.tar.gz");
    let target = voc_dir.join("orbvoc.txt");

    println!("cargo:rerun-if-changed={}", archive.display());

    if !archive.exists() {
        // Nothing to unpack (e.g. a checkout without the archive). Don't fail
        // the build — the binary will surface a clear error if it actually
        // tries to load the vocabulary.
        return;
    }

    if target.exists() {
        return;
    }

    if let Err(e) = unpack(&archive, &voc_dir) {
        // Emit a cargo warning rather than aborting — the user can still
        // unpack manually with `tar -xzf`.
        println!(
            "cargo:warning=failed to unpack {}: {}",
            archive.display(),
            e
        );
    }
}

fn unpack(archive: &Path, dest_dir: &Path) -> std::io::Result<()> {
    let f = fs::File::open(archive)?;
    let gz = flate2::read::GzDecoder::new(f);
    let mut ar = tar::Archive::new(gz);
    ar.unpack(dest_dir)?;

    // The upstream archive contains `ORBvoc.txt` (capitalized) but the rest
    // of the codebase, docs, and CLI examples use the lowercase name. Rename
    // for consistency.
    let upstream = dest_dir.join("ORBvoc.txt");
    let canonical = dest_dir.join("orbvoc.txt");
    if upstream.exists() {
        fs::rename(&upstream, &canonical)?;
    }
    Ok(())
}

/// Parse the leading `major.minor` of a `pkg-config --modversion` string.
fn parse_major_minor(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().and_then(|m| m.parse().ok()).unwrap_or(0);
    Some((major, minor))
}
