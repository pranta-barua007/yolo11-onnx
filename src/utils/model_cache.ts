/**
 * Browser-side ONNX model cache.
 *
 * Uses the Cache API to store model ArrayBuffers and localStorage
 * for custom model metadata. Works in both main thread and Web Workers.
 *
 * No DOM dependencies — fully reusable and non-blocking.
 */

const CACHE_NAME = "yolo-model-cache-v1";
const CUSTOM_MODELS_KEY = "yolo-custom-models";

// ── Cache API helpers (model bytes) ──

/** Check if a model exists in cache and return its ArrayBuffer. */
export async function getModelFromCache(key: string): Promise<ArrayBuffer | null> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(key);
    if (!response) return null;
    return response.arrayBuffer();
  } catch {
    return null;
  }
}

/** Store a model ArrayBuffer in the cache. */
export async function putModelInCache(key: string, buffer: ArrayBuffer): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const response = new Response(buffer, {
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Cached-At": new Date().toISOString(),
      },
    });
    await cache.put(key, response);
  } catch (err) {
    console.warn("[model_cache] Failed to cache model:", err);
  }
}

/** Remove a specific model from the cache. */
export async function deleteModelFromCache(key: string): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME);
    await cache.delete(key);
  } catch (err) {
    console.warn("[model_cache] Failed to delete cached model:", err);
  }
}

/** Wipe the entire model cache. */
export async function clearModelCache(): Promise<void> {
  try {
    await caches.delete(CACHE_NAME);
  } catch (err) {
    console.warn("[model_cache] Failed to clear cache:", err);
  }
}

// ── localStorage helpers (custom model metadata) ──

export interface CachedCustomModel {
  name: string;
  classes: string[];
  /** Cache key used to retrieve bytes from Cache API. */
  cacheKey: string;
}

/** Load persisted custom model metadata from localStorage. */
export function getCustomModelsMetadata(): CachedCustomModel[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const raw = localStorage.getItem(CUSTOM_MODELS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/** Save custom model metadata to localStorage. */
export function setCustomModelsMetadata(models: CachedCustomModel[]): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(models));
  } catch (err) {
    console.warn("[model_cache] Failed to save custom models metadata:", err);
  }
}

/** Add a single custom model's metadata (deduplicates by name). */
export function addCustomModelMetadata(model: CachedCustomModel): void {
  const existing = getCustomModelsMetadata();
  const filtered = existing.filter((m) => m.cacheKey !== model.cacheKey);
  filtered.push(model);
  setCustomModelsMetadata(filtered);
}

/** Remove a custom model's metadata by cache key. */
export function removeCustomModelMetadata(cacheKey: string): void {
  const existing = getCustomModelsMetadata();
  setCustomModelsMetadata(existing.filter((m) => m.cacheKey !== cacheKey));
}
