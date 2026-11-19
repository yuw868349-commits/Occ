//! Removing a directory when the process ends.
//!
//! A test that compiles a fixture needs somewhere to put it, and the natural
//! place is the system temporary directory. The trouble is that a program
//! that exits without cleaning up leaves that directory behind, and a suite
//! that runs often fills `/tmp` with copies of every fixture it built. The
//! copies are small but they are never reclaimed, and a machine that has run
//! the suite a thousand times has a thousand of them.
//!
//! Rust has no hook for "when the process exits", and the usual answer —
//! dropping a value that owns the directory — does not work for a value held
//! in a `static`, because statics are never dropped. The C runtime does have
//! such a hook, [`atexit`], and it runs while the process is still
//! single-threaded and the filesystem is still there, which is exactly when a
//! directory can be removed.
//!
//! [`atexit`]: https://man7.org/linux/man-pages/man3/atexit.3.html

use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The directories to remove when the process ends.
static PENDING: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// A closure that can run at exit.
///
/// It takes no arguments and returns nothing, which is what `atexit` wants.
type AtExit = extern "C" fn();

unsafe extern "C" {
    fn atexit(f: AtExit) -> i32;
}

/// Remove `dir` when the process ends.
///
/// The removal happens once, at exit. A caller that wants the directory gone
/// sooner removes it itself; this is the safety net for the case where the
/// caller did not get the chance, which is what a test that panics looks like
/// from the outside.
///
/// Removing a directory that is already gone is not an error: the point of
/// the call is that the directory is not there afterwards, and it is not.
pub fn remove_dir_at_exit(dir: &Path) {
    // A poisoned lock means a panic happened while the list was held. The
    // data inside is still valid — the panic did not corrupt it — so the
    // directory is recorded and the hook registered regardless. Leaving a
    // directory behind is the failure this function exists to prevent.
    match PENDING.lock() {
        Ok(mut pending) => pending.push(dir.to_path_buf()),
        Err(poisoned) => poisoned.into_inner().push(dir.to_path_buf()),
    }
    register();
}

/// Whether the cleanup hook has been registered already.
///
/// The C runtime calls the hook once, so it must only be registered once.
static REGISTERED: Mutex<bool> = Mutex::new(false);

/// Register the exit hook, once.
fn register() {
    let mut registered = match REGISTERED.lock() {
        Ok(r) => r,
        Err(poisoned) => poisoned.into_inner(),
    };
    if *registered {
        return;
    }
    *registered = true;
    // SAFETY: `cleanup` is an `extern "C" fn` with no arguments and no
    // return value, which is the signature `atexit` requires. It touches only
    // a mutex and the filesystem, both of which are usable at exit because
    // `atexit` handlers run before the process is torn down.
    unsafe {
        atexit(cleanup);
    }
}

/// The hook `atexit` calls.
extern "C" fn cleanup() {
    let pending = match PENDING.lock() {
        Ok(mut p) => std::mem::take(&mut *p),
        Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
    };
    remove_all(pending);
}

/// Remove every directory in `dirs`.
///
/// A directory that is not there is not an error: the point of the call is
/// that the directory is not there afterwards, and it is not.
fn remove_all(dirs: Vec<PathBuf>) {
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that drain the pending list.
    ///
    /// The list is global, and a test that empties it would take another
    /// test's directory with it. The lock makes the two tests that call the
    /// hook run one after the other.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A directory under the system temporary directory, named so that two
    /// tests running at the same time do not collide.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("occ-sys-cleanup-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create the scratch directory");
        dir
    }

    #[test]
    fn a_registered_directory_is_removed() {
        let _guard = SERIAL.lock().expect("the lock is not poisoned");
        let dir = scratch("removed");
        std::fs::write(dir.join("file"), b"x").expect("write a file");
        remove_dir_at_exit(&dir);
        assert!(
            dir.exists(),
            "the directory is still there before the hook runs"
        );
        cleanup();
        assert!(
            !dir.exists(),
            "the directory must be gone after the hook runs"
        );
    }

    #[test]
    fn a_directory_that_is_already_gone_is_not_an_error() {
        let _guard = SERIAL.lock().expect("the lock is not poisoned");
        let dir = scratch("gone");
        remove_dir_at_exit(&dir);
        std::fs::remove_dir_all(&dir).expect("remove it by hand");
        // The hook must not fail on a directory that is not there: the point
        // of the call is that the directory is not there afterwards.
        cleanup();
        assert!(!dir.exists());
    }

    #[test]
    fn removing_a_list_of_directories_removes_all_of_them() {
        let a = scratch("list-a");
        let b = scratch("list-b");
        remove_all(vec![a.clone(), b.clone()]);
        assert!(!a.exists() && !b.exists());
    }
}
