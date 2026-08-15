use crate::cached_object_store::stats::CachedObjectStoreStats;
use crate::cached_object_store::storage_fs::FsCacheStorage;
use crate::cached_object_store::LocalCacheEntry;
use crate::config::ObjectStoreCacheOptions;
use crate::rand::DbRand;
use bytes::{Bytes, BytesMut};
use futures::{future::BoxFuture, stream, stream::BoxStream, StreamExt};
use object_store::{path::Path, GetOptions, GetResult, ObjectMeta, ObjectStore};
use object_store::{Attributes, GetRange, GetResultPayload, PutMultipartOptions, PutResult};
use object_store::{ListResult, MultipartUpload, PutOptions, PutPayload};
use slatedb_common::clock::SystemClock;
use std::collections::HashMap;
use std::{ops::Range, sync::Arc};
use tokio::sync::OnceCell;

use crate::cached_object_store::admission::AdmissionPicker;
use crate::cached_object_store::storage::{LocalCacheStorage, PartID};
use crate::error::SlateDBError;
use log::warn;

use crate::utils::build_concurrent;
use slatedb_common::metrics::MetricsRecorderHelper;

#[derive(Debug, Clone)]
pub(crate) struct CachedObjectStore {
    object_store: Arc<dyn ObjectStore>,
    pub(crate) part_size_bytes: usize, // expected to be aligned with mb or kb
    pub(crate) cache_storage: Arc<dyn LocalCacheStorage>,
    pub(crate) admission_picker: AdmissionPicker,
    pub(crate) cache_puts: bool,
    // Absolute path of the root folder relative to the bucket. See #1319.
    resolved_root: Arc<OnceCell<Path>>,
    stats: Arc<CachedObjectStoreStats>,
    /// In-memory cache of head metadata, keyed by upstream location.
    /// Populated only on write paths (L0 flush via `cached_put_opts`,
    /// compaction via the tee commit). Reads never fill this map - that
    /// keeps the cache coherent with the on-disk head files without an
    /// invalidation protocol, and avoids amplifying read-side memory
    /// pressure when reads stream over many short-lived objects.
    ///
    /// Hits here skip both the on-disk head read and any S3 HEAD round
    /// trip. The per-entry payload is ~hundreds of bytes; an SST working
    /// set in the thousands of entries fits comfortably in low MBs.
    head_cache: Arc<parking_lot::Mutex<HashMap<Path, (ObjectMeta, Attributes)>>>,
}

impl CachedObjectStore {
    pub(crate) fn new(
        object_store: Arc<dyn ObjectStore>,
        cache_storage: Arc<dyn LocalCacheStorage>,
        part_size_bytes: usize,
        cache_puts: bool,
        stats: Arc<CachedObjectStoreStats>,
    ) -> Result<Arc<Self>, SlateDBError> {
        if part_size_bytes == 0 || !part_size_bytes.is_multiple_of(1024) {
            return Err(SlateDBError::InvalidCachePartSize);
        }

        Ok(Arc::new(Self {
            object_store,
            part_size_bytes,
            cache_storage,
            stats,
            admission_picker: AdmissionPicker::default(),
            cache_puts,
            resolved_root: Arc::new(OnceCell::new()),
            head_cache: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        }))
    }

    /// Insert (or overwrite) an entry in the in-memory head cache.
    ///
    /// Callers:
    /// - Write paths (`cached_put_opts` after the on-disk save_head
    ///   succeeds; the tee-commit path after a streaming write
    ///   finalizes).
    /// - Read paths, when a head is served from the on-disk cache for
    ///   a *compacted-SST path*. Those paths are immutable under a
    ///   stable id, so the in-memory copy can't diverge from the
    ///   authoritative source. Non-SST paths (manifests, WALs) must
    ///   NOT trigger this from the read side - they can be rewritten
    ///   without any invalidation signal into this cache.
    pub(crate) fn populate_head_cache(&self, location: &Path, meta: ObjectMeta, attrs: Attributes) {
        self.head_cache
            .lock()
            .insert(location.clone(), (meta, attrs));
    }

    /// Drop an entry from the in-memory head cache. Used when the
    /// underlying object is removed so the next reader doesn't see
    /// stale metadata. Best-effort; absence of an entry is fine.
    pub(crate) fn invalidate_head_cache(&self, location: &Path) {
        self.head_cache.lock().remove(location);
    }

    pub(crate) async fn start_evictor(&self) {
        self.cache_storage.start_evictor().await;
    }

    /// Begin a streaming tee write for `location`. Returns `None` if the
    /// cache root has not yet been resolved (no successful GET/HEAD has
    /// happened) or the underlying storage refuses tee writes. Callers fall
    /// back to lazy cache population on the first read.
    pub(crate) fn begin_tee(
        &self,
        location: &Path,
    ) -> Option<Box<dyn crate::cached_object_store::LocalCacheTee>> {
        let cache_location = self.cache_location_for(location)?;
        self.cache_storage
            .begin_tee(&cache_location, self.part_size_bytes)
    }

    /// Remove cached parts and head metadata for `location`, if any. This is
    /// best-effort and never fails the caller: errors are logged. If the
    /// cache root has not yet been resolved (no GET/HEAD has succeeded for
    /// any object), this is a no-op since the cache cannot hold any entry
    /// for this location.
    ///
    /// Currently unused by production code: SST deletion in `TableStore::
    /// delete_sst` deliberately leaves cache entries in place so a stale
    /// reader snapshot can keep serving (the LRU evictor reclaims space
    /// later). Kept available for tests and any future caller that
    /// genuinely wants synchronous invalidation.
    #[allow(dead_code)]
    pub(crate) async fn invalidate(&self, location: &Path) {
        // Drop the in-memory head first so a concurrent reader can't
        // pick up stale metadata while we tear down the disk entry.
        self.invalidate_head_cache(location);
        let Some(cache_location) = self.cache_location_for(location) else {
            return;
        };
        if let Err(e) = self.cache_storage.remove(&cache_location).await {
            warn!(
                "failed to invalidate cache entry [location={}, error={:?}]",
                location, e
            );
        }
    }

    /// Build a `CachedObjectStore` from `ObjectStoreCacheOptions`, returning `None`
    /// if caching is not configured (i.e. `root_folder` is `None`). When `Some` is
    /// returned the evictor has already been started.
    pub(crate) async fn from_config(
        object_store: Arc<dyn ObjectStore>,
        options: &ObjectStoreCacheOptions,
        recorder: &MetricsRecorderHelper,
        clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
    ) -> Result<Option<Arc<Self>>, SlateDBError> {
        let cache_root_folder = match &options.root_folder {
            None => return Ok(None),
            Some(f) => f,
        };
        let stats = Arc::new(CachedObjectStoreStats::new(recorder));
        let cache_storage: Arc<dyn crate::cached_object_store::LocalCacheStorage> = if options
            .use_io_uring
        {
            #[cfg(target_os = "linux")]
            {
                match crate::cached_object_store::storage_uring::IoUringCacheStorage::try_new(
                    cache_root_folder.clone(),
                    options.direct_io,
                ) {
                    Ok(s) => Arc::new(s),
                    Err(e) => {
                        warn!(
                                "io_uring cache storage init failed; falling back to FsCacheStorage [error={:?}]",
                                e
                            );
                        Arc::new(FsCacheStorage::new(
                            cache_root_folder.clone(),
                            options.max_cache_size_bytes,
                            options.scan_interval,
                            stats.clone(),
                            clock.clone(),
                            rand.clone(),
                            options.max_open_file_handles,
                            options.direct_io,
                        ))
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                warn!("use_io_uring set but not on Linux; falling back to FsCacheStorage");
                Arc::new(FsCacheStorage::new(
                    cache_root_folder.clone(),
                    options.max_cache_size_bytes,
                    options.scan_interval,
                    stats.clone(),
                    clock.clone(),
                    rand.clone(),
                    options.max_open_file_handles,
                    options.direct_io,
                ))
            }
        } else {
            Arc::new(FsCacheStorage::new(
                cache_root_folder.clone(),
                options.max_cache_size_bytes,
                options.scan_interval,
                stats.clone(),
                clock,
                rand,
                options.max_open_file_handles,
                options.direct_io,
            ))
        };
        let cached = Self::new(
            object_store,
            cache_storage,
            options.part_size_bytes,
            options.cache_puts,
            stats,
        )?;
        cached.start_evictor().await;
        Ok(Some(cached))
    }

    /// Load files into cache up to a maximum number of bytes.
    /// This method fetches objects from the provided paths and stores them in the cache
    /// until the specified max_bytes limit is reached.
    pub(crate) async fn load_files_to_cache(
        &self,
        file_paths: Vec<Path>,
        max_bytes: usize,
    ) -> Result<(), SlateDBError> {
        if file_paths.is_empty() || max_bytes == 0 {
            return Ok(());
        }

        let mut remaining_bytes = max_bytes;
        let mut files_to_load = Vec::with_capacity(file_paths.len());

        // First pass: sequentially fetch metadata for each SST and
        // pick the ones that fit in the budget. Routed through
        // `cached_head` so that, on a warm-disk restart, the head can
        // be served from the on-disk cache instead of issuing a fresh
        // S3 HEAD per SST. Cold starts still hit S3 once per SST
        // (same as before), but the result is now saved to disk by
        // `cached_head`'s miss path, which keeps the second pass's
        // `maybe_prefetch_range` from doing it again.
        for path in file_paths {
            match self.cached_head(&path).await {
                Ok(meta) => {
                    let file_size = meta.size as usize;
                    if remaining_bytes >= file_size {
                        remaining_bytes -= file_size;
                        files_to_load.push(path);
                    } else {
                        // We can't fit this file, so we stop here
                        break;
                    }
                }
                Err(e) => {
                    // If file doesn't exist or can't be accessed, we stop here
                    warn!("Failed to preload all SSTs to cache: {:?}", e);
                    break;
                }
            }
        }

        // Second pass: load the selected files in bounded parallelism and
        // cache them. This must not go through `maybe_prefetch_range`: the
        // metadata pass above populated the in-memory and on-disk head
        // caches, and `maybe_prefetch_range` short-circuits on a head hit
        // before fetching any parts, which would make the preload a no-op
        // on a cold cache.
        let degree_of_parallelism = 32;
        let _result = build_concurrent(files_to_load.into_iter(), degree_of_parallelism, |path| {
            let this = self.clone();
            async move {
                match this.prefetch_missing_parts(&path).await {
                    Ok(_) => Ok(Some(())),
                    Err(e) => {
                        warn!(
                            "Failed to prefetch file into cache [path={}, error={:?}]",
                            path, e
                        );
                        Ok(None) // best-effort: skip errors
                    }
                }
            }
        })
        .await;

        Ok(())
    }

    /// Fetch every part of `location` that is not already in the disk
    /// cache, coalescing contiguous gaps into single aligned ranged GETs.
    /// Head lookups may be served from the head caches; part existence is
    /// checked against the cache storage directly.
    async fn prefetch_missing_parts(&self, location: &Path) -> object_store::Result<()> {
        let meta = self.cached_head(location).await?;
        let Some(cache_location) = self.cache_location_for(location) else {
            return Ok(());
        };
        let entry = self
            .cache_storage
            .entry(&cache_location, self.part_size_bytes);
        let part_size = self.part_size_bytes as u64;
        let total_parts = usize::try_from(meta.size.div_ceil(part_size))
            .expect("part count exceeds usize");
        let cached: std::collections::HashSet<PartID> =
            entry.cached_parts().await?.into_iter().collect();

        let mut idx = 0;
        while idx < total_parts {
            if cached.contains(&idx) {
                idx += 1;
                continue;
            }
            let gap_start = idx;
            while idx < total_parts && !cached.contains(&idx) {
                idx += 1;
            }
            let byte_range =
                (gap_start as u64 * part_size)..std::cmp::min(idx as u64 * part_size, meta.size);
            let opts = GetOptions {
                range: Some(GetRange::Bounded(byte_range)),
                ..GetOptions::default()
            };
            let get_result = self.object_store.get_opts(location, opts).await?;
            if self.resolve_root(location, &get_result.meta.location) {
                // Disk-full and similar save errors are non-fatal for a
                // prefetch; reads fall back to the object store.
                self.save_get_result(get_result).await.ok();
            }
        }
        Ok(())
    }

    /// Returns the canonical cache key for a requested location.
    ///
    /// The key is `resolved_root + location` once root resolution succeeds.
    /// Returns `None` while the root is still unresolved. The root is resolved
    /// lazily from observed metadata locations, so this method may return `None`
    /// for early requests until successful GET/HEAD responses are observed.
    fn cache_location_for(&self, location: &Path) -> Option<Path> {
        self.resolved_root.get().map(|root| {
            if root.as_ref().is_empty() {
                return location.clone();
            }
            root.parts().chain(location.parts()).collect()
        })
    }

    /// Lazily resolves the root prefix and validates the derived cache key.
    ///
    /// ## Arguments
    ///
    /// - `requested_location`: the location from the incoming request, treated as a suffix
    ///   for root inference.
    /// - `meta_location`: the location from the observed metadata, expected to be
    ///   `root + requested_location`.
    ///
    /// ## Returns
    ///
    /// Returns `true` only when:
    ///
    /// - `resolved_root` is already known or can be safely inferred from metadata; and
    /// - the derived canonical cache key matches `meta_location`.
    ///
    /// Returns `false` otherwise. This is a defensive guard: on mismatch, cache writes are
    /// skipped to avoid poisoning cache entries with unsafe keys.
    fn resolve_root(&self, requested_location: &Path, meta_location: &Path) -> bool {
        // If root is not resolved yet, try to infer it from the metadata location
        if self.resolved_root.get().is_none() {
            let Some(root) = Self::infer_root(requested_location, meta_location) else {
                warn!(
                    "failed to resolve cache root lazily [requested_location={}, meta_location={}]",
                    requested_location, meta_location,
                );
                return false;
            };
            let _ = self.resolved_root.set(root);
        }
        // Get cache location so we can verify it matches the metadata location. This should always
        // succeed after root resolution.
        let Some(cache_location) = self.cache_location_for(requested_location) else {
            warn!(
                "cache location is unexpectedly unavailable after root resolution [requested_location={}, meta_location={}]",
                requested_location, meta_location,
            );
            return false;
        };
        // Verify the cache location matches the metadata location. Again, should always be true, but
        // you can never trust object stores completely.
        if &cache_location != meta_location {
            warn!(
                "resolved root mismatch [requested_location={}, cache_location={}, meta_location={}]",
                requested_location, cache_location, meta_location,
            );
            return false;
        }
        true
    }

    /// Infers a root prefix by treating `requested_location` as a suffix of `meta_location`.
    ///
    /// Returns `Some(root)` when `meta_location == root + requested_location`,
    /// otherwise returns `None`.
    fn infer_root(requested_location: &Path, meta_location: &Path) -> Option<Path> {
        let requested_str = requested_location.as_ref();
        let meta_str = meta_location.as_ref();

        if requested_str.is_empty() {
            return Some(meta_location.clone());
        }

        let prefix = meta_str.strip_suffix(requested_str)?;

        // Ensure suffix matching happens at a path-segment boundary.
        if !prefix.is_empty() && !prefix.ends_with('/') {
            return None;
        }

        Some(Path::from(prefix.trim_end_matches('/')))
    }

    pub(crate) async fn cached_head(&self, location: &Path) -> object_store::Result<ObjectMeta> {
        let cacheable = Self::is_compacted_sst_path(location);
        if cacheable {
            self.stats.object_store_cache_head_access.increment(1);
        }
        // In-memory shortcut: populated only on write commits, so a hit
        // here is authoritative without any disk I/O.
        if let Some((meta, _)) = self.head_cache.lock().get(location).cloned() {
            if cacheable {
                self.stats.object_store_cache_head_hits.increment(1);
            }
            return Ok(meta);
        }
        if let Some(cache_location) = self.cache_location_for(location) {
            let entry = self
                .cache_storage
                .entry(&cache_location, self.part_size_bytes);
            if let Ok(Some((meta, attrs))) = entry.read_head().await {
                if cacheable {
                    self.stats.object_store_cache_head_hits.increment(1);
                    // Populate the in-memory head cache from the disk
                    // hit. Safe for compacted-SST paths because they
                    // are immutable under a stable id (the write side
                    // never rewrites the same path). Skipped for non-
                    // cacheable paths (manifests, WALs) which can be
                    // rewritten and have no invalidation channel into
                    // this cache.
                    self.populate_head_cache(location, meta.clone(), attrs);
                }
                return Ok(meta);
            }
        }

        // Cache miss: fall through to S3 HEAD. This path has
        // multi-second tail latency on AWS; the (access - hit) gap
        // tracked here is the most useful early-warning signal for
        // reader stalls during compaction events.
        let result = self
            .object_store
            .get_opts(
                location,
                GetOptions {
                    range: None,
                    head: true,
                    ..Default::default()
                },
            )
            .await?;
        let meta = result.meta.clone();
        if self.resolve_root(location, &meta.location) {
            self.save_get_result(result).await.ok();
        }
        Ok(meta)
    }

    pub(crate) async fn cached_get_opts(
        &self,
        location: &Path,
        opts: GetOptions,
    ) -> object_store::Result<GetResult> {
        let (meta, attributes) = self.maybe_prefetch_range(location, opts.clone()).await?;

        // If we still can't derive a safe canonical cache key after calling
        // maybe_prefetch_range, bypass cache for this request since there's no
        // point in fetching by range.
        if self.cache_location_for(location).is_none() {
            return self.object_store.get_opts(location, opts.clone()).await;
        }

        let get_range = opts.range.clone();
        let range = self.canonicalize_range(get_range, meta.size)?;
        let parts = self.split_range_into_parts(range.clone());

        // read parts, and concatenate them into a single stream. please note that
        // some of these part may not be cached, we'll still fallback to the object
        // store to get the missing parts.
        let futures = parts
            .into_iter()
            .map(|(part_id, range_in_part)| self.read_part(location, part_id, range_in_part))
            .collect::<Vec<_>>();
        let result_stream = stream::iter(futures).then(|fut| fut).boxed();

        Ok(GetResult {
            meta,
            range,
            attributes,
            payload: GetResultPayload::Stream(result_stream),
        })
    }

    async fn cached_put_opts(
        &self,
        location: &Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<PutResult> {
        // Only cache if the cache_puts option is enabled, AND the put is for
        // a compacted SST. WAL frames, manifests, and compactions metadata
        // are excluded:
        //   - WAL: only used for crash recovery; caching it wastes disk and
        //     write bandwidth on data we never re-read on the hot path.
        //   - manifests / compactions metadata: small, mutable, and read
        //     through dedicated paths that don't benefit from this layer.
        // Compacted SSTs are the only PUT-shaped writes worth caching here.
        // Streaming SST writes (multipart uploads) flow through the prewarm
        // tee in EncodedSsTableWriter, not through this method.
        if !self.cache_puts || !Self::is_compacted_sst_path(location) {
            return self.object_store.put_opts(location, payload, opts).await;
        }

        // Compute the payload size before consuming `payload` for the
        // upstream PUT - we need it to build the ObjectMeta we save
        // alongside the parts so subsequent reads don't have to do an
        // S3 HEAD round-trip.
        let payload_size: u64 = payload.iter().map(|b| b.len() as u64).sum();

        // First, write to the upstream object store (cloning payload is cheap since it's just a Arc internally)
        let result = self
            .object_store
            .put_opts(location, payload.clone(), opts)
            .await?;

        // Build the head metadata from the PUT result. We construct it
        // *before* the cache-root resolution check so the in-memory head
        // cache can be populated unconditionally - readers of this
        // location will get an in-memory hit even if the on-disk cache
        // was unavailable at write time (e.g. root not yet resolved on
        // the first writer-side put).
        //
        // last_modified is a stub: the cache doesn't use this
        // field for invalidation, only echoes it back from
        // cached_head(). The upstream object's real last_modified
        // is filled in lazily the next time a HEAD goes upstream.
        let head_meta = ObjectMeta {
            location: location.clone(),
            last_modified: chrono::DateTime::<chrono::Utc>::MIN_UTC,
            size: payload_size,
            e_tag: result.e_tag.clone(),
            version: result.version.clone(),
        };
        let head_attrs = object_store::Attributes::new();
        // Pre-populate the in-memory head cache: this entry is now the
        // authoritative head for this location. Done before the on-disk
        // cache checks below so a startup race where `resolved_root` is
        // still unset cannot leave the head unrecorded - the next reader
        // hits the in-memory entry instead of S3 HEAD.
        self.populate_head_cache(location, head_meta.clone(), head_attrs.clone());

        // Then, save to local cache if admission policy allows it
        let Some(cache_location) = self.cache_location_for(location) else {
            return Ok(result);
        };
        let entry = self
            .cache_storage
            .entry(&cache_location, self.part_size_bytes);
        if self.admission_picker.pick(entry.as_ref()).admitted() {
            // Save the head metadata to the on-disk cache. The
            // in-memory head was already populated above; here we
            // mirror it on disk so a reader served after eviction
            // of the in-memory entry still hits the local cache.
            entry.save_head((&head_meta, &head_attrs)).await.ok();

            // Convert PutPayload to stream and save parts to cache.
            let stream = stream::iter(payload.into_iter()).map(Ok::<Bytes, object_store::Error>);

            // Save parts only, ignoring errors (cache failures shouldn't fail the PUT operation)
            self.save_parts_stream(entry, stream, 0).await.ok();
        }

        Ok(result)
    }

    /// True when `location` points at a compacted-SST file. We match on a
    /// `compacted/` path segment rather than any specific root prefix so the
    /// detection works under prefix-stores, multi-tenant layouts, and tests
    /// that mount the DB under arbitrary paths.
    fn is_compacted_sst_path(location: &Path) -> bool {
        location.parts().any(|p| p.as_ref() == "compacted")
    }

    // if an object is not cached before, maybe_prefetch_range will try to prefetch the object from the
    // object store and save the parts into the local disk cache. the prefetching is helpful to reduce the
    // number of GET requests to the object store, it'll try to aggregate the parts among the range into a
    // single GET request, and save the related parts into local disks together.
    // when it sends GET requests to the object store, the range is expected to be ALIGNED with the part
    // size.
    async fn maybe_prefetch_range(
        &self,
        location: &Path,
        mut opts: GetOptions,
    ) -> object_store::Result<(ObjectMeta, Attributes)> {
        // Mirror `cached_head`: every call to this function is a head
        // lookup, so it must count toward head_access_count regardless
        // of whether the cache root is resolved. Without this, the
        // counter only reflects calls that came through `cached_head`
        // directly, and `cached_get_opts` traffic is invisible.
        let cacheable = Self::is_compacted_sst_path(location);
        if cacheable {
            self.stats.object_store_cache_head_access.increment(1);
        }
        // In-memory shortcut: populated only on write commits, so a hit
        // here is authoritative without any disk I/O.
        if let Some((meta, attrs)) = self.head_cache.lock().get(location).cloned() {
            if cacheable {
                self.stats.object_store_cache_head_hits.increment(1);
            }
            return Ok((meta, attrs));
        }
        if let Some(cache_location) = self.cache_location_for(location) {
            let entry = self
                .cache_storage
                .entry(&cache_location, self.part_size_bytes);
            match entry.read_head().await {
                Ok(Some((meta, attrs))) => {
                    if cacheable {
                        self.stats.object_store_cache_head_hits.increment(1);
                        // Populate the in-memory head cache from the
                        // disk hit. See the matching block in
                        // `cached_head` for the safety argument: only
                        // valid for immutable compacted-SST paths.
                        self.populate_head_cache(location, meta.clone(), attrs.clone());
                    }
                    return Ok((meta, attrs));
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(
                        "failed to read head from disk cache, will fallback to object store [location={}, error={:?}]",
                        location, e,
                    );
                }
            }
        }

        if let Some(range) = &opts.range {
            opts.range = Some(self.align_get_range(range));
        }

        let get_result = self.object_store.get_opts(location, opts).await?;
        let result_meta = get_result.meta.clone();
        let result_attrs = get_result.attributes.clone();
        // swallow the error on saving to disk here (the disk might be already full), just fallback
        // to the object store.
        // TODO: add a warning log here.
        if self.resolve_root(location, &result_meta.location) {
            self.save_get_result(get_result).await.ok();
        }
        Ok((result_meta, result_attrs))
    }

    /// save the GetResult to the disk cache, a GetResult may be transformed into multiple part
    /// files and a meta file. please note that the `range` in the GetResult is expected to be
    /// aligned with the part size.
    async fn save_get_result(&self, result: GetResult) -> object_store::Result<u64> {
        let part_size_bytes_u64 = self.part_size_bytes as u64;
        assert!(result.range.start.is_multiple_of(part_size_bytes_u64));
        assert!(
            result.range.end.is_multiple_of(part_size_bytes_u64)
                || result.range.end == result.meta.size
        );

        let entry = self
            .cache_storage
            .entry(&result.meta.location, self.part_size_bytes);
        let object_size = result.meta.size;

        if self.admission_picker.pick(entry.as_ref()).admitted() {
            entry.save_head((&result.meta, &result.attributes)).await?;

            let start_part_number = usize::try_from(result.range.start / part_size_bytes_u64)
                .expect("Part number exceeds u32 on a 32-bit system. Try increasing part size.");

            let stream = result.into_stream();

            self.save_parts_stream(entry, stream, start_part_number)
                .await?;
        }

        Ok(object_size)
    }

    /// Save a stream of bytes to cache as parts, starting from the specified part number.
    /// Returns the number of bytes saved.
    /// This method only saves the data parts - the head should be saved separately.
    async fn save_parts_stream<S>(
        &self,
        entry: Box<dyn LocalCacheEntry>,
        mut stream: S,
        start_part_number: usize,
    ) -> object_store::Result<usize>
    where
        S: stream::Stream<Item = Result<Bytes, object_store::Error>> + Unpin,
    {
        let mut buffer = BytesMut::new();
        let mut part_number = start_part_number;
        let mut total_bytes: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            total_bytes += chunk.len();
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= self.part_size_bytes {
                let to_write = buffer.split_to(self.part_size_bytes);
                entry.save_part(part_number, to_write.into()).await?;
                part_number += 1;
            }
        }

        // Save any remaining bytes as the last part
        if !buffer.is_empty() {
            entry.save_part(part_number, buffer.into()).await?;
        }

        Ok(total_bytes)
    }

    // split the range into parts, and return the part id and the range inside the part.
    fn split_range_into_parts(&self, range: Range<u64>) -> Vec<(PartID, Range<usize>)> {
        let part_size_bytes_u64 = self.part_size_bytes as u64;
        let range_aligned = self.align_range(&range, self.part_size_bytes);
        let start_part = range_aligned.start / part_size_bytes_u64;
        let end_part = range_aligned.end / part_size_bytes_u64;
        let mut parts: Vec<_> = (start_part..end_part)
            .map(|part_id| {
                (
                    usize::try_from(part_id).expect("Number of parts exceeds usize"),
                    Range {
                        start: 0,
                        end: self.part_size_bytes,
                    },
                )
            })
            .collect();
        if parts.is_empty() {
            return vec![];
        }
        if let Some(first_part) = parts.first_mut() {
            first_part.1.start = usize::try_from(range.start % part_size_bytes_u64)
                .expect("Part size is too large to fit in a usize");
        }
        if let Some(last_part) = parts.last_mut() {
            if !range.end.is_multiple_of(part_size_bytes_u64) {
                last_part.1.end = usize::try_from(range.end % part_size_bytes_u64)
                    .expect("Part size is too large to fit in a usize");
            }
        }
        parts
    }

    /// get from disk if the parts are cached, otherwise start a new GET request.
    /// the io errors on reading the disk caches will be ignored, just fallback to
    /// the object store.
    fn read_part(
        &self,
        location: &Path,
        part_id: PartID,
        range_in_part: Range<usize>,
    ) -> BoxFuture<'static, object_store::Result<Bytes>> {
        let this = self.clone();
        let location = location.clone();
        Box::pin(async move {
            this.stats.object_store_cache_part_access.increment(1);

            // Try local cache first.
            if let Some(cache_location) = this.cache_location_for(&location) {
                let entry = this
                    .cache_storage
                    .entry(&cache_location, this.part_size_bytes);
                // Cache hit, so return immediately.
                if let Ok(Some(bytes)) = entry.read_part(part_id, range_in_part.clone()).await {
                    this.stats.object_store_cache_part_hits.increment(1);
                    return Ok(bytes);
                }
            }

            // Cache miss, so we need to fetch from the object store.
            let part_range = Range {
                start: (part_id * this.part_size_bytes) as u64,
                end: ((part_id + 1) * this.part_size_bytes) as u64,
            };
            let get_result = this
                .object_store
                .get_opts(
                    &location,
                    GetOptions {
                        range: Some(GetRange::Bounded(part_range)),
                        ..Default::default()
                    },
                )
                .await?;

            // Get the cache entry again after successful get so we can cache the part.
            let cache_entry = if this.resolve_root(&location, &get_result.meta.location) {
                this.cache_location_for(&location).map(|cache_location| {
                    this.cache_storage
                        .entry(&cache_location, this.part_size_bytes)
                })
            } else {
                // If the root resolution fails, we won't be able to derive a canonical cache
                // key. Skip saving to cache to avoid poisoning the cache with unsafe keys.
                None
            };

            // Save the head and the part to cache for future accesses. Just read the bytes
            // if we still can't derive a canonical cache key.
            let bytes = if let Some(entry) = cache_entry {
                // Save the head and the part to cache for future accesses.
                let meta = get_result.meta.clone();
                let attrs = get_result.attributes.clone();
                let bytes = get_result.bytes().await?;
                entry.save_head((&meta, &attrs)).await.ok();
                entry.save_part(part_id, bytes.clone()).await.ok();
                bytes
            } else {
                get_result.bytes().await?
            };

            Ok(Bytes::copy_from_slice(&bytes.slice(range_in_part)))
        })
    }

    // given the range and object size, return the canonicalized `Range<usize>` with concrete start and
    // end.
    fn canonicalize_range(
        &self,
        range: Option<GetRange>,
        object_size: u64,
    ) -> object_store::Result<Range<u64>> {
        let (start_offset, end_offset) = match range {
            None => (0, object_size),
            Some(range) => match range {
                GetRange::Bounded(range) => {
                    if range.start >= object_size {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::StartTooLarge {
                                requested: range.start,
                                length: object_size,
                            }),
                        });
                    }
                    if range.start >= range.end {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::Inconsistent {
                                start: range.start,
                                end: range.end,
                            }),
                        });
                    }
                    (range.start, range.end.min(object_size))
                }
                GetRange::Offset(offset) => {
                    if offset >= object_size {
                        return Err(object_store::Error::Generic {
                            store: "cached_object_store",
                            source: Box::new(InvalidGetRange::StartTooLarge {
                                requested: offset,
                                length: object_size,
                            }),
                        });
                    }
                    (offset, object_size)
                }
                GetRange::Suffix(suffix) => (object_size.saturating_sub(suffix), object_size),
            },
        };
        Ok(Range {
            start: start_offset,
            end: end_offset,
        })
    }

    fn align_get_range(&self, range: &GetRange) -> GetRange {
        match range {
            GetRange::Bounded(bounded) => {
                let aligned = self.align_range(bounded, self.part_size_bytes);
                GetRange::Bounded(aligned)
            }
            GetRange::Suffix(suffix) => {
                let suffix_aligned = self.align_range(&(0..*suffix), self.part_size_bytes).end;
                GetRange::Suffix(suffix_aligned)
            }
            GetRange::Offset(offset) => {
                let offset_aligned = *offset - *offset % self.part_size_bytes as u64;
                GetRange::Offset(offset_aligned)
            }
        }
    }

    fn align_range(&self, range: &Range<u64>, alignment: usize) -> Range<u64> {
        let alignment = alignment as u64;
        let start_aligned = range.start - range.start % alignment;
        let end_aligned = range.end.div_ceil(alignment) * alignment;
        Range {
            start: start_aligned,
            end: end_aligned,
        }
    }
}

impl std::fmt::Display for CachedObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CachedObjectStore({}, {})",
            self.object_store, self.cache_storage
        )
    }
}

#[async_trait::async_trait]
impl ObjectStore for CachedObjectStore {
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.cached_get_opts(location, options).await
    }

    async fn head(&self, location: &Path) -> object_store::Result<ObjectMeta> {
        self.cached_head(location).await
    }

    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.cached_put_opts(location, payload, opts).await
    }

    async fn put_multipart(
        &self,
        location: &Path,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.object_store.put_multipart(location).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.object_store.put_multipart_opts(location, opts).await
    }

    async fn delete(&self, location: &Path) -> object_store::Result<()> {
        // TODO: handle cache eviction
        // Drop the in-memory head so a subsequent reader doesn't see
        // metadata describing an object the upstream no longer has.
        // The on-disk cache eviction is still a TODO above; this is
        // strictly an improvement over the previous behavior.
        self.invalidate_head_cache(location);
        self.object_store.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.object_store.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.object_store.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.object_store.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.object_store.copy(from, to).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.object_store.rename(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.object_store.copy_if_not_exists(from, to).await
    }

    async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.object_store.rename_if_not_exists(from, to).await
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum InvalidGetRange {
    #[error("Range start too large, requested: {requested}, length: {length}")]
    StartTooLarge { requested: u64, length: u64 },

    #[error("Range started at {start} and ended at {end}")]
    Inconsistent { start: u64, end: u64 },
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use object_store::{path::Path, GetOptions, GetRange, ObjectStore, PutPayload};
    use rand::Rng;

    use super::CachedObjectStore;
    use crate::cached_object_store::stats::CachedObjectStoreStats;
    use crate::cached_object_store::storage::{LocalCacheStorage, PartID};
    use crate::cached_object_store::storage_fs::FsCacheEntry;
    use crate::cached_object_store::storage_fs::FsCacheStorage;
    use crate::rand::DbRand;
    use crate::test_utils::gen_rand_bytes;
    use slatedb_common::clock::DefaultSystemClock;
    use slatedb_common::metrics::MetricsRecorderHelper;

    fn new_test_cache_folder() -> std::path::PathBuf {
        let mut rng = rand::rng();
        let dir_name: String = (0..10)
            .map(|_| rng.sample(rand::distr::Alphanumeric) as char)
            .collect();
        let path = format!("/tmp/testcache-{}", dir_name);
        let _ = std::fs::remove_dir_all(&path);
        std::path::PathBuf::from(path)
    }

    #[derive(Debug)]
    struct MismatchedMetaStore {
        inner: Arc<dyn ObjectStore>,
        bad_location: Path,
    }

    impl MismatchedMetaStore {
        fn new(inner: Arc<dyn ObjectStore>, bad_location: Path) -> Self {
            Self {
                inner,
                bad_location,
            }
        }
    }

    impl std::fmt::Display for MismatchedMetaStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MismatchedMetaStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for MismatchedMetaStore {
        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            let mut result = self.inner.get_opts(location, options).await?;
            result.meta.location = self.bad_location.clone();
            Ok(result)
        }

        async fn head(&self, location: &Path) -> object_store::Result<object_store::ObjectMeta> {
            self.inner.head(location).await
        }

        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart(
            &self,
            location: &Path,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart(location).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn delete(&self, location: &Path) -> object_store::Result<()> {
            self.inner.delete(location).await
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy(&self, from: &Path, to: &Path) -> object_store::Result<()> {
            self.inner.copy(from, to).await
        }

        async fn rename(&self, from: &Path, to: &Path) -> object_store::Result<()> {
            self.inner.rename(from, to).await
        }

        async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
            self.inner.copy_if_not_exists(from, to).await
        }

        async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
            self.inner.rename_if_not_exists(from, to).await
        }
    }

    #[test]
    fn test_infer_root() {
        assert_eq!(
            CachedObjectStore::infer_root(
                &Path::from("manifest/0001.manifest"),
                &Path::from("tenant-a/manifest/0001.manifest")
            ),
            Some(Path::from("tenant-a"))
        );
        assert_eq!(
            CachedObjectStore::infer_root(
                &Path::from("manifest/0001.manifest"),
                &Path::from("other/path")
            ),
            None
        );
        assert_eq!(
            CachedObjectStore::infer_root(&Path::from("b/c"), &Path::from("ab/c")),
            None
        );
    }

    #[tokio::test]
    async fn test_lazy_resolve_root_from_meta_location() -> object_store::Result<()> {
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let prefixed: Arc<dyn ObjectStore> = Arc::new(object_store::prefix::PrefixStore::new(
            backing_store.clone(),
            Path::from("tenant-a"),
        ));
        let cached_store =
            CachedObjectStore::new(prefixed, cache_storage, 1024, false, stats).unwrap();

        let relative_location = Path::from("manifest/0001.manifest");
        let full_location = Path::from("tenant-a/manifest/0001.manifest");
        let payload = Bytes::from_static(b"tenant-a-manifest");
        backing_store
            .put(&full_location, PutPayload::from_bytes(payload.clone()))
            .await?;

        assert_eq!(cached_store.resolved_root.get().cloned(), None);
        let got = cached_store
            .cached_get_opts(&relative_location, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got, payload);
        assert_eq!(
            cached_store.resolved_root.get().cloned(),
            Some(Path::from("tenant-a"))
        );

        let scoped_entry = cached_store.cache_storage.entry(&full_location, 1024);
        assert_eq!(scoped_entry.cached_parts().await?.len(), 1);
        let unscoped_entry = cached_store.cache_storage.entry(&relative_location, 1024);
        assert_eq!(unscoped_entry.cached_parts().await?.len(), 0);

        backing_store.delete(&full_location).await?;
        let got_cached = cached_store
            .cached_get_opts(&relative_location, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got_cached, payload);
        Ok(())
    }

    #[tokio::test]
    async fn test_shared_cache_with_prefix_stores_does_not_collide() -> object_store::Result<()> {
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            {
                let recorder = MetricsRecorderHelper::noop();
                Arc::new(CachedObjectStoreStats::new(&recorder))
            },
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));

        let store_a: Arc<dyn ObjectStore> = Arc::new(object_store::prefix::PrefixStore::new(
            backing_store.clone(),
            Path::from("db-a"),
        ));
        let store_b: Arc<dyn ObjectStore> = Arc::new(object_store::prefix::PrefixStore::new(
            backing_store.clone(),
            Path::from("db-b"),
        ));

        let cached_a = CachedObjectStore::new(store_a, cache_storage.clone(), 1024, false, {
            let recorder = MetricsRecorderHelper::noop();
            Arc::new(CachedObjectStoreStats::new(&recorder))
        })
        .unwrap();
        let cached_b = CachedObjectStore::new(store_b, cache_storage.clone(), 1024, false, {
            let recorder = MetricsRecorderHelper::noop();
            Arc::new(CachedObjectStoreStats::new(&recorder))
        })
        .unwrap();

        let relative = Path::from("manifest/0001.manifest");
        let full_a = Path::from("db-a/manifest/0001.manifest");
        let full_b = Path::from("db-b/manifest/0001.manifest");
        let payload_a = Bytes::from_static(b"tenant-a-data");
        let payload_b = Bytes::from_static(b"tenant-b-data");

        backing_store
            .put(&full_a, PutPayload::from_bytes(payload_a.clone()))
            .await?;
        backing_store
            .put(&full_b, PutPayload::from_bytes(payload_b.clone()))
            .await?;

        let got_a = cached_a
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        let got_b = cached_b
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got_a, payload_a);
        assert_eq!(got_b, payload_b);
        assert_eq!(
            cached_a.resolved_root.get().cloned(),
            Some(Path::from("db-a"))
        );
        assert_eq!(
            cached_b.resolved_root.get().cloned(),
            Some(Path::from("db-b"))
        );

        let unscoped_entry = cache_storage.entry(&relative, 1024);
        assert_eq!(unscoped_entry.cached_parts().await?.len(), 0);
        let scoped_a = cache_storage.entry(&full_a, 1024);
        let scoped_b = cache_storage.entry(&full_b, 1024);
        assert_eq!(scoped_a.cached_parts().await?.len(), 1);
        assert_eq!(scoped_b.cached_parts().await?.len(), 1);

        backing_store.delete(&full_a).await?;
        backing_store.delete(&full_b).await?;
        let got_cached_a = cached_a
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        let got_cached_b = cached_b
            .cached_get_opts(&relative, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got_cached_a, payload_a);
        assert_eq!(got_cached_b, payload_b);
        Ok(())
    }

    #[tokio::test]
    async fn test_meta_location_mismatch_bypasses_cache() -> object_store::Result<()> {
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let bad_meta_store: Arc<dyn ObjectStore> = Arc::new(MismatchedMetaStore::new(
            backing_store.clone(),
            Path::from("wrong/root"),
        ));
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let cached_store =
            CachedObjectStore::new(bad_meta_store, cache_storage, 1024, false, stats).unwrap();

        let location = Path::from("data/file.sst");
        let payload = Bytes::from_static(b"payload");
        backing_store
            .put(&location, PutPayload::from_bytes(payload.clone()))
            .await?;

        assert_eq!(cached_store.resolved_root.get().cloned(), None);
        let got = cached_store
            .cached_get_opts(&location, GetOptions::default())
            .await?
            .bytes()
            .await?;
        assert_eq!(got, payload);
        assert_eq!(cached_store.resolved_root.get().cloned(), None);
        let entry = cached_store.cache_storage.entry(&location, 1024);
        assert_eq!(entry.cached_parts().await?.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_save_result_not_aligned() -> object_store::Result<()> {
        let payload = gen_rand_bytes(1024 * 3 + 32);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        object_store
            .put(
                &Path::from("/data/testfile1"),
                PutPayload::from_bytes(payload.clone()),
            )
            .await?;
        let location = Path::from("/data/testfile1");
        let get_result = object_store.get(&location).await?;

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));

        let part_size = 1024;
        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, part_size, false, stats)
                .unwrap();
        let entry = cached_store.cache_storage.entry(&location, 1024);

        let object_size_hint = cached_store.save_get_result(get_result).await?;
        assert_eq!(object_size_hint, 1024 * 3 + 32);

        // assert the cached meta
        let head = entry.read_head().await?;
        assert_eq!(head.unwrap().0.size, 1024 * 3 + 32);

        // assert the parts
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 4);
        assert_eq!(
            entry.read_part(0, 0..part_size).await?,
            Some(payload.slice(0..1024))
        );
        assert_eq!(
            entry.read_part(1, 0..part_size).await?,
            Some(payload.slice(1024..2048))
        );
        assert_eq!(
            entry.read_part(2, 0..part_size).await?,
            Some(payload.slice(2048..3072))
        );
        // check that the unaligned part was also cached
        assert_eq!(
            entry.read_part(3, 0..32).await?,
            Some(payload.slice(3072..3104))
        );

        // delete part 2, known_cache_size is still known
        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 2, 1024);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(2, 0..part_size).await?, None);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts, vec![0, 1, 3]);

        // delete part 3, known_cache_size become None
        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 3, 1024);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(3, 0..part_size).await?, None);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts, vec![0, 1]);
        Ok(())
    }

    #[tokio::test]
    async fn test_save_result_aligned() -> object_store::Result<()> {
        let payload = gen_rand_bytes(1024 * 3);
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        object_store
            .put(
                &Path::from("/data/testfile1"),
                PutPayload::from_bytes(payload.clone()),
            )
            .await?;
        let location = Path::from("/data/testfile1");
        let get_result = object_store.get(&location).await?;
        let part_size = 1024;

        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));

        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, part_size, false, stats).unwrap();
        let entry = cached_store.cache_storage.entry(&location, part_size);
        let object_size_hint = cached_store.save_get_result(get_result).await?;
        assert_eq!(object_size_hint, 1024 * 3);
        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 3);
        assert_eq!(
            entry.read_part(0, 0..part_size).await?,
            Some(payload.slice(0..1024))
        );
        assert_eq!(
            entry.read_part(1, 0..part_size).await?,
            Some(payload.slice(1024..2048))
        );
        assert_eq!(
            entry.read_part(2, 0..part_size).await?,
            Some(payload.slice(2048..3072))
        );

        let evict_part_path =
            FsCacheEntry::make_part_path(test_cache_folder.clone(), &location, 2, part_size);
        std::fs::remove_file(evict_part_path).unwrap();
        assert_eq!(entry.read_part(2, 0..part_size).await?, None);

        let cached_parts = entry.cached_parts().await?;
        assert_eq!(cached_parts.len(), 2);
        Ok(())
    }

    #[test]
    fn test_split_range_into_parts() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));

        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        struct Test {
            input: (Option<GetRange>, usize),
            expect: Vec<(PartID, std::ops::Range<usize>)>,
        }
        let tests = [
            Test {
                input: (None, 1024 * 3),
                expect: vec![(0, 0..1024), (1, 0..1024), (2, 0..1024)],
            },
            Test {
                input: (None, 1024 * 3 + 12),
                expect: vec![(0, 0..1024), (1, 0..1024), (2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (None, 12),
                expect: vec![(0, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(0..1024)), 1024),
                expect: vec![(0, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024)), 20000),
                expect: vec![(0, 128..1024)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024 + 12)), 20000),
                expect: vec![(0, 128..1024), (1, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(128..1024 * 2 + 12)), 20000),
                expect: vec![(0, 128..1024), (1, 0..1024), (2, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(1024 * 2..1024 * 3 + 12)), 200000),
                expect: vec![(2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (Some(GetRange::Bounded(1024 * 2 - 2..1024 * 3 + 12)), 20000),
                expect: vec![(1, 1022..1024), (2, 0..1024), (3, 0..12)],
            },
            Test {
                input: (Some(GetRange::Suffix(128)), 1024),
                expect: vec![(0, 896..1024)],
            },
            Test {
                input: (Some(GetRange::Suffix(1024 * 2 + 8)), 1024 * 4),
                expect: vec![(1, 1016..1024), (2, 0..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(8)), 1024 * 4),
                expect: vec![(0, 8..1024), (1, 0..1024), (2, 0..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(1024 * 2 + 8)), 1024 * 4),
                expect: vec![(2, 8..1024), (3, 0..1024)],
            },
            Test {
                input: (Some(GetRange::Offset(1024 * 2 + 8)), 1024 * 4 + 2),
                expect: vec![(2, 8..1024), (3, 0..1024), (4, 0..2)],
            },
        ];

        for t in tests.iter() {
            let range = cached_store
                .canonicalize_range(t.input.0.clone(), t.input.1 as u64)
                .unwrap();
            let parts = cached_store.split_range_into_parts(range);
            assert_eq!(parts, t.expect, "input: {:?}", t.input);
        }
    }

    #[test]
    fn test_align_range() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        let aligned = cached_store.align_range(&(9..1025), 1024);
        assert_eq!(aligned, 0..2048);
        let aligned = cached_store.align_range(&(1024 + 1..2048 + 4), 1024);
        assert_eq!(aligned, 1024..3072);
    }

    #[test]
    fn test_align_get_range() {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder,
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let cached_store =
            CachedObjectStore::new(object_store, cache_storage, 1024, false, stats).unwrap();

        let aligned = cached_store.align_get_range(&GetRange::Bounded(9..1025));
        assert_eq!(aligned, GetRange::Bounded(0..2048));
        let aligned = cached_store.align_get_range(&GetRange::Bounded(9..2048));
        assert_eq!(aligned, GetRange::Bounded(0..2048));
        let aligned = cached_store.align_get_range(&GetRange::Suffix(12));
        assert_eq!(aligned, GetRange::Suffix(1024));
        let aligned = cached_store.align_get_range(&GetRange::Suffix(1024));
        assert_eq!(aligned, GetRange::Suffix(1024));
        let aligned = cached_store.align_get_range(&GetRange::Offset(1024));
        assert_eq!(aligned, GetRange::Offset(1024));
        let aligned = cached_store.align_get_range(&GetRange::Offset(12));
        assert_eq!(aligned, GetRange::Offset(0));
    }

    #[tokio::test]
    async fn test_cached_object_store_impl_object_store() -> object_store::Result<()> {
        let object_store = Arc::new(object_store::memory::InMemory::new());
        let test_cache_folder = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            test_cache_folder.clone(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, 1024, false, stats)
                .unwrap();

        let test_path = Path::from("/data/testdata1");
        let test_payload = gen_rand_bytes(1024 * 3 + 2);
        object_store
            .put(&test_path, PutPayload::from_bytes(test_payload.clone()))
            .await?;

        // test get entire object
        let test_ranges = vec![
            Some(GetRange::Offset(260817)),
            None,
            Some(GetRange::Bounded(1000..2048)),
            Some(GetRange::Bounded(1000..260817)),
            Some(GetRange::Suffix(10)),
            Some(GetRange::Suffix(260817)),
            Some(GetRange::Offset(1000)),
            Some(GetRange::Offset(0)),
            Some(GetRange::Offset(1028)),
            Some(GetRange::Offset(260817)),
            Some(GetRange::Offset(1024 * 3 + 2)),
            Some(GetRange::Offset(1024 * 3 + 1)),
            #[allow(clippy::reversed_empty_ranges)]
            Some(GetRange::Bounded(2900..2048)),
            Some(GetRange::Bounded(10..10)),
        ];

        // test get a range
        for range in test_ranges.iter() {
            let want = object_store
                .get_opts(
                    &test_path,
                    GetOptions {
                        range: range.clone(),
                        ..Default::default()
                    },
                )
                .await;
            let got = cached_store
                .cached_get_opts(
                    &test_path,
                    GetOptions {
                        range: range.clone(),
                        ..Default::default()
                    },
                )
                .await;
            match (want, got) {
                (Ok(want), Ok(got)) => {
                    assert_eq!(want.range, got.range);
                    assert_eq!(want.meta, got.meta);
                    assert_eq!(want.bytes().await?, got.bytes().await?);
                }
                (Err(want), Err(got)) => {
                    if want.to_string().to_lowercase().contains("range") {
                        assert!(got.to_string().to_lowercase().contains("range"));
                    }
                }
                (origin_result, cached_result) => {
                    panic!("expect: {:?}, got: {:?}", origin_result, cached_result);
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_preload_cache() {
        let cache_dir = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            cache_dir,
            Some(10 * 1024 * 1024), // 10MB
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));

        let object_store = Arc::new(object_store::memory::InMemory::new());

        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, 1024, true, stats).unwrap();

        // Create some test files to preload
        let test_paths = vec![
            Path::from("file1.sst"),
            Path::from("file2.sst"),
            Path::from("file3.sst"),
        ];

        let test_data = gen_rand_bytes(2048); // 2KB per file

        // Put test files in object store
        for path in &test_paths {
            object_store
                .put(path, PutPayload::from_bytes(test_data.clone()))
                .await
                .unwrap();
        }

        // Test preloading with max cache size
        cached_store
            .load_files_to_cache(test_paths.clone(), 10 * 1024) // 10KB limit
            .await
            .unwrap();

        // Verify that files are cached by checking if we can read from cache
        for path in &test_paths {
            let entry = cached_store.cache_storage.entry(path, 1024);
            let cached_parts = entry.cached_parts().await.unwrap();
            assert_eq!(cached_parts.len(), 2); // 2KB = 2 parts of 1024 bytes
        }
    }

    #[tokio::test]
    async fn test_preload_cache_above_limit() {
        let cache_dir = new_test_cache_folder();
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            cache_dir,
            Some(10 * 1024 * 1024), // 10MB
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));

        let object_store = Arc::new(object_store::memory::InMemory::new());

        let cached_store =
            CachedObjectStore::new(object_store.clone(), cache_storage, 1024, true, stats).unwrap();

        // Create some test files
        let test_paths = vec![Path::from("file1.sst"), Path::from("file2.sst")];

        let test_data = gen_rand_bytes(2048); // 2KB per file

        // Put test files in object store
        for path in &test_paths {
            object_store
                .put(path, PutPayload::from_bytes(test_data.clone()))
                .await
                .unwrap();
        }

        // Test load_files_to_cache with 0 bytes limit (should load nothing)
        cached_store
            .load_files_to_cache(test_paths.clone(), 0)
            .await
            .unwrap();

        // Verify that files are NOT cached since preloading was disabled
        for path in &test_paths {
            let entry = cached_store.cache_storage.entry(path, 1024);
            let cached_parts = entry.cached_parts().await.unwrap();
            assert_eq!(cached_parts.len(), 0); // No parts should be cached
        }
    }

    #[tokio::test]
    async fn test_invalidate_removes_cached_parts_and_head() {
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let cached_store =
            CachedObjectStore::new(backing_store.clone(), cache_storage, 1024, false, stats)
                .unwrap();

        // Seed the upstream object and pull it through the cache so the cache
        // has both parts and a head entry, plus a resolved root.
        let location = Path::from("compacted/abc.sst");
        let payload = gen_rand_bytes(2500); // > one part, < two parts
        backing_store
            .put(&location, PutPayload::from_bytes(payload.clone()))
            .await
            .unwrap();
        let _ = cached_store
            .cached_get_opts(&location, GetOptions::default())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        let entry = cached_store.cache_storage.entry(&location, 1024);
        assert!(
            !entry.cached_parts().await.unwrap().is_empty(),
            "expected cache to be populated after a GET"
        );
        assert!(
            entry.read_head().await.unwrap().is_some(),
            "expected head metadata to be cached"
        );

        // Invalidate and confirm the per-object directory is empty.
        cached_store.invalidate(&location).await;
        let entry = cached_store.cache_storage.entry(&location, 1024);
        assert!(
            entry.cached_parts().await.unwrap().is_empty(),
            "cache parts should be gone after invalidate"
        );
        assert!(
            entry.read_head().await.unwrap().is_none(),
            "head should be gone after invalidate"
        );
    }

    #[tokio::test]
    async fn test_cache_puts_skips_wal_and_manifest_paths() {
        // cache_puts=true should only populate the cache for compacted-SST
        // paths. WAL frames and manifest writes go straight to upstream so
        // we don't burn local disk on bytes we'll never re-read on the hot
        // path.
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let cached =
            CachedObjectStore::new(backing_store.clone(), cache_storage, 1024, true, stats)
                .unwrap();

        // Resolve the cache root by reading something through the cache
        // (otherwise cache_location_for returns None and the cache writes
        // skip for an unrelated reason).
        let warm = Path::from("warmup-object");
        backing_store
            .put(&warm, PutPayload::from_bytes(Bytes::from_static(b"hi")))
            .await
            .unwrap();
        let _ = cached.cached_head(&warm).await.unwrap();

        // Put to a WAL path. The upstream should accept it but the cache
        // should not have any parts for it.
        let wal_path = Path::from("db/wal/00000000000000000001.sst");
        cached
            .put(
                &wal_path,
                PutPayload::from_bytes(Bytes::from_static(b"wal-frame")),
            )
            .await
            .unwrap();
        let wal_entry = cached.cache_storage.entry(&wal_path, 1024);
        assert!(
            wal_entry.cached_parts().await.unwrap().is_empty(),
            "WAL puts must not populate the cache"
        );

        // Put to a manifest path. Same expectation.
        let manifest_path = Path::from("db/manifest/0001.manifest");
        cached
            .put(
                &manifest_path,
                PutPayload::from_bytes(Bytes::from_static(b"manifest-bytes")),
            )
            .await
            .unwrap();
        let manifest_entry = cached.cache_storage.entry(&manifest_path, 1024);
        assert!(
            manifest_entry.cached_parts().await.unwrap().is_empty(),
            "manifest puts must not populate the cache"
        );

        // Put to a compacted SST path. This one *should* land in the cache.
        let compacted_path = Path::from("db/compacted/01HQ.sst");
        cached
            .put(
                &compacted_path,
                PutPayload::from_bytes(Bytes::from_static(b"compacted-sst-bytes")),
            )
            .await
            .unwrap();
        let compacted_entry = cached.cache_storage.entry(&compacted_path, 1024);
        assert!(
            !compacted_entry.cached_parts().await.unwrap().is_empty(),
            "compacted SST puts must populate the cache when cache_puts=true"
        );
    }

    #[tokio::test]
    async fn test_invalidate_unresolved_root_is_noop() {
        // Before any successful GET/HEAD, the cache root is unresolved and the
        // cache cannot hold entries for any location. invalidate() must not
        // panic or error in this state.
        let backing_store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());
        let recorder = MetricsRecorderHelper::noop();
        let stats = Arc::new(CachedObjectStoreStats::new(&recorder));
        let cache_storage = Arc::new(FsCacheStorage::new(
            new_test_cache_folder(),
            None,
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        ));
        let cached_store =
            CachedObjectStore::new(backing_store, cache_storage, 1024, false, stats).unwrap();

        cached_store
            .invalidate(&Path::from("compacted/never-fetched.sst"))
            .await;
    }
}
