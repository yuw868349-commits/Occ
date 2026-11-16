//! Read-only memory mapping without the `memmap2` crate.
//!
//! Mapping rather than reading matters for large binaries: a 400 MB shared
//! library should not be copied into the server's heap just to read its section
//! headers, and the kernel's page cache already holds the file.
//!
//! The trade is that a mapped file must not change size while it is mapped.
//! Touching a page past the new end raises `SIGBUS`, which kills the process
//! outright — there is no error to handle. [`Mapped::map_file`] therefore
//! records the length at map time and the caller is documented to treat the
//! mapping as invalid if the file changes. For an analysis tool this is
//! acceptable: the input is a fixed artefact, not a log being appended to.

use crate::errno::{self, SysError};
use std::ffi::{c_int, c_long, c_void};
use std::fs::File;
use std::io;
use std::ops::Deref;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

const PROT_READ: c_int = 1;
const MAP_PRIVATE: c_int = 2;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: c_long,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn sysconf(name: c_int) -> c_long;
}

const SC_PAGESIZE: c_int = 30;

/// System page size, which is the alignment `mmap` requires for file offsets.
pub fn page_size() -> usize {
    // SAFETY: `sysconf` with a valid name cannot fault.
    let v = unsafe { sysconf(SC_PAGESIZE) };
    if v > 0 { v as usize } else { 4096 }
}

/// A read-only mapping of a file.
pub struct Mapped {
    ptr: *mut u8,
    len: usize,
    path: PathBuf,
}

// SAFETY: the mapping is read-only for its whole lifetime and the kernel keeps
// the pages valid until `Drop` unmaps them. Sharing the slice across threads is
// therefore no different from sharing a `&[u8]`.
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl Mapped {
    /// Map `path` read-only.
    ///
    /// An empty file maps to an empty slice without calling `mmap`, because
    /// mapping zero bytes fails with `EINVAL` and every caller would otherwise
    /// have to special-case it.
    pub fn map_file(path: &Path) -> Result<Self, SysError> {
        let file = File::open(path).map_err(|e| SysError {
            op: "open",
            code: e.raw_os_error().unwrap_or(errno::EIO),
        })?;
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        if len == 0 {
            return Ok(Mapped {
                ptr: std::ptr::null_mut(),
                len: 0,
                path: path.to_path_buf(),
            });
        }
        let len = usize::try_from(len).map_err(|_| SysError {
            op: "mmap",
            code: 12, // ENOMEM
        })?;
        // SAFETY: `addr` is null (let the kernel choose), `len` is the real file
        // size, and the fd stays open for the duration of the call. `mmap`
        // keeps its own reference, so closing the `File` at the end of this
        // function is fine.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr as isize == -1 {
            return Err(SysError::now("mmap"));
        }
        Ok(Mapped {
            ptr: ptr.cast::<u8>(),
            len,
            path: path.to_path_buf(),
        })
    }

    /// Number of mapped bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the file was empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Path this mapping came from.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Deref for Mapped {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` addresses `len` readable bytes, mapped for the lifetime
        // of `self`, and is never written through.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: `ptr` and `len` are exactly what `mmap` returned and accepted.
        unsafe {
            munmap(self.ptr.cast::<c_void>(), self.len);
        }
    }
}

impl std::fmt::Debug for Mapped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapped")
            .field("path", &self.path)
            .field("len", &self.len)
            .finish()
    }
}

/// Read a whole file into memory, with the path attached to any error.
///
/// Used below the mapping threshold and for `/proc` files, which cannot be
/// mapped usefully because their reported size is zero.
pub fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn page_size_is_a_power_of_two() {
        let p = page_size();
        assert!(p.is_power_of_two(), "{p}");
        assert!(p >= 4096);
    }

    #[test]
    fn maps_an_existing_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("occ_sys_map_test.bin");
        {
            let mut f = File::create(&path).expect("create");
            f.write_all(b"hello mmap").expect("write");
        }
        let m = Mapped::map_file(&path).expect("map");
        assert_eq!(&m[..], b"hello mmap");
        assert_eq!(m.len(), 10);
        assert!(!m.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn maps_an_empty_file_to_an_empty_slice() {
        let dir = std::env::temp_dir();
        let path = dir.join("occ_sys_empty_test.bin");
        File::create(&path).expect("create");
        let m = Mapped::map_file(&path).expect("map");
        assert!(m.is_empty());
        assert_eq!(&m[..], b"");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_reports_open_not_mmap() {
        let err = Mapped::map_file(Path::new("/nonexistent/occ/probe")).unwrap_err();
        assert_eq!(err.op, "open");
        assert_eq!(err.name(), "ENOENT");
    }
}
