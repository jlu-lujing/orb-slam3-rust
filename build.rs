// Build script.
//
// Currently just unpacks the ORB vocabulary if needed. The vocabulary is
// shipped compressed (`vocabulary/orbvoc.txt.tar.gz`, ~40 MiB) and consumed
// at runtime as plain text (`vocabulary/orbvoc.txt`, ~140 MiB). We unpack
// once and re-unpack only when the archive is newer than the extracted file.

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
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
