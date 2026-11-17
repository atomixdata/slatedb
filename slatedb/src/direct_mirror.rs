//! Prototype: reads mirrored SST blocks through concrete types.
//!
//! A block read normally goes through `ReadOnlyObject`, then the
//! `RetryingObjectStore`, `InstrumentedObjectStore`, and `ObjectStoreMirror`
//! trait objects, the mirror policy's route, and the mirror's `dyn Vfs`. With
//! a direct mirror, a `TableStore` reads the local file of a mirrored SST with
//! a concrete `MirrorHandle` and `StdVfs`. A block that is not local, or fails
//! to decode, takes the normal path, which also handles the retry of a
//! corrupt block.

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path;
use slatedb_common::vfs::StdVfs;
use slatedb_mirror::{LocalFile, MirrorHandle};

use crate::blob::ReadOnlyBlob;
use crate::error::SlateDBError;

/// A mirror and the VFS of its local files.
pub struct DirectMirror {
    handle: MirrorHandle,
    vfs: StdVfs,
}

static DIRECT_MIRROR: parking_lot::Mutex<Option<Arc<DirectMirror>>> =
    parking_lot::const_mutex(None);

/// Makes every `TableStore` built after this call read mirrored blocks
/// through `handle` and `vfs`, which must be the VFS of the mirror. `None`
/// turns direct reads off for stores built after the call.
pub fn set_direct_mirror(mirror: Option<(MirrorHandle, StdVfs)>) {
    *DIRECT_MIRROR.lock() = mirror.map(|(handle, vfs)| Arc::new(DirectMirror { handle, vfs }));
}

/// The direct mirror for a `TableStore` built now.
pub(crate) fn current() -> Option<Arc<DirectMirror>> {
    DIRECT_MIRROR.lock().clone()
}

impl DirectMirror {
    /// The local copy of an SST. The first call for a slot asks the mirror
    /// for the file of `path()` and keeps it in the slot.
    ///
    /// The mirror lookup takes a lock that every read shares. If the mirror
    /// later evicts or deletes the copy, it marks the kept file removed. Reads
    /// of the SST then take the normal path, until a new handle replaces this
    /// one.
    pub(crate) fn local_blob<'a>(
        &'a self,
        pinned: &'a crate::db_state::pinning::PinnedSstMetadata,
        path: impl FnOnce() -> Path,
    ) -> Option<LocalBlob<'a>> {
        if pinned.local_file.get().is_none() {
            let _ = pinned.local_file.set(self.handle.local_file(&path())?);
        }
        let file = pinned.local_file.get()?;
        if file.is_removed() {
            return None;
        }
        Some(LocalBlob {
            vfs: &self.vfs,
            file,
        })
    }
}

/// The local file of one mirrored SST.
pub(crate) struct LocalBlob<'a> {
    vfs: &'a StdVfs,
    file: &'a LocalFile,
}

impl ReadOnlyBlob for LocalBlob<'_> {
    async fn len(&self) -> Result<u64, SlateDBError> {
        Ok(self.file.size())
    }

    async fn read_range(&self, range: Range<u64>) -> Result<Bytes, SlateDBError> {
        static FIRST_READ: std::sync::Once = std::sync::Once::new();
        // Shows in bench output that the direct path is in use.
        #[allow(clippy::disallowed_macros)]
        FIRST_READ.call_once(|| eprintln!("direct mirror: first direct block read"));
        self.vfs
            .read_range_direct(self.file.path(), range)
            .await
            .map_err(SlateDBError::from)
    }

    async fn read(&self) -> Result<Bytes, SlateDBError> {
        self.read_range(0..self.file.size()).await
    }
}
