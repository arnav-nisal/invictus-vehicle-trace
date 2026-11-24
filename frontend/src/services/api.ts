import {
  BoxesData,
  JobStatusResponse,
  Stage1Result,
  VerificationResult,
  VerifyRequest,
} from '../types';

/**
 * 1. POST /api/videos/select-local
 * Directly triggers Stage 1 tracking or retrieves cached result from FastAPI backend.
 */
export async function selectLocalVideo(
  filename: string,
  forceRefresh = false,
  signal?: AbortSignal
): Promise<JobStatusResponse> {
  const res = await fetch('/api/videos/select-local', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, force_refresh: forceRefresh }),
    signal,
  });

  const contentType = res.headers.get('content-type') || '';

  if (!res.ok) {
    let detail = `Server responded with ${res.status}: ${res.statusText}`;
    if (contentType.includes('application/json')) {
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) {
          detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        // Non-JSON error payload
      }
    }
    throw new Error(detail);
  }

  if (!contentType.includes('application/json')) {
    throw new Error(`Expected JSON response from select-local, but received ${contentType || 'non-JSON'}`);
  }

  const data = await res.json();
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid JSON payload returned from /api/videos/select-local');
  }

  return data as JobStatusResponse;
}

/**
 * 2. GET /api/videos/{job_id}/status
 * Polls Stage 1 execution status from FastAPI backend.
 */
export async function getJobStatus(
  jobId: string,
  signal?: AbortSignal
): Promise<JobStatusResponse> {
  if (!jobId || typeof jobId !== 'string' || jobId.trim() === '' || jobId === 'undefined') {
    throw new Error(`Invalid jobId provided to getJobStatus: "${String(jobId)}"`);
  }

  const res = await fetch(`/api/videos/${encodeURIComponent(jobId)}/status`, {
    signal,
  });

  const contentType = res.headers.get('content-type') || '';

  if (!res.ok) {
    let detail = `Status request failed (${res.status}: ${res.statusText})`;
    if (contentType.includes('application/json')) {
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) {
          detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        // Non-JSON error payload
      }
    }
    throw new Error(detail);
  }

  if (!contentType.includes('application/json')) {
    throw new Error(`Expected JSON response from status endpoint, but received ${contentType || 'non-JSON'}`);
  }

  const data = await res.json();
  if (!data || typeof data !== 'object') {
    throw new Error(`Invalid JSON payload returned from status for job "${jobId}"`);
  }

  return data as JobStatusResponse;
}

/**
 * 3. GET boxes from boxes_url directly from backend / outputs.
 */
export async function fetchBoxesData(
  boxesUrl: string,
  signal?: AbortSignal
): Promise<BoxesData> {
  if (!boxesUrl || typeof boxesUrl !== 'string') {
    throw new Error(`Invalid boxesUrl provided: "${String(boxesUrl)}"`);
  }

  const res = await fetch(boxesUrl, { signal });
  const contentType = res.headers.get('content-type') || '';

  if (!res.ok) {
    throw new Error(`Failed to load bounding boxes from ${boxesUrl} (${res.status} ${res.statusText})`);
  }

  if (!contentType.includes('application/json')) {
    throw new Error(`Expected JSON bounding boxes from ${boxesUrl}, but received ${contentType || 'non-JSON'}`);
  }

  const data = await res.json();
  if (!data || typeof data !== 'object' || !data.frames || typeof data.frames !== 'object') {
    throw new Error(`Invalid bounding boxes payload received from ${boxesUrl}`);
  }

  return data as BoxesData;
}

/**
 * 4. Polling helper: Polls GET /api/videos/{job_id}/status every 700ms
 * Cancels cleanly via AbortSignal to avoid React dev mode double-mount leaks.
 */
export async function pollStage1Job(
  jobId: string,
  onProgress?: (statusResp: JobStatusResponse) => void,
  signal?: AbortSignal
): Promise<Stage1Result> {
  if (!jobId || typeof jobId !== 'string' || jobId.trim() === '' || jobId === 'undefined') {
    throw new Error(`Cannot poll Stage 1 job with invalid jobId: "${String(jobId)}"`);
  }

  if (signal?.aborted) {
    throw new DOMException('Polling aborted before start', 'AbortError');
  }

  // First immediate status check
  const initial = await getJobStatus(jobId, signal);
  if (signal?.aborted) {
    throw new DOMException('Polling aborted', 'AbortError');
  }

  if (onProgress) {
    onProgress(initial);
  }

  if (initial.status === 'ready' && initial.result) {
    return initial.result;
  }
  if (initial.status === 'error') {
    throw new Error(initial.error || initial.message || 'Stage 1 processing failed on server');
  }

  return new Promise<Stage1Result>((resolve, reject) => {
    let interval: ReturnType<typeof setInterval> | null = null;

    const cleanup = () => {
      if (interval !== null) {
        clearInterval(interval);
        interval = null;
      }
      if (signal) {
        signal.removeEventListener('abort', handleAbort);
      }
    };

    const handleAbort = () => {
      cleanup();
      reject(new DOMException('Polling aborted', 'AbortError'));
    };

    if (signal) {
      signal.addEventListener('abort', handleAbort, { once: true });
    }

    interval = setInterval(async () => {
      if (signal?.aborted) {
        cleanup();
        reject(new DOMException('Polling aborted', 'AbortError'));
        return;
      }

      try {
        const resp = await getJobStatus(jobId, signal);

        if (signal?.aborted) {
          cleanup();
          return;
        }

        if (onProgress) {
          onProgress(resp);
        }

        if (resp.status === 'ready' && resp.result) {
          cleanup();
          resolve(resp.result);
        } else if (resp.status === 'error') {
          cleanup();
          reject(new Error(resp.error || resp.message || 'Stage 1 processing failed on server'));
        }
      } catch (err) {
        cleanup();
        reject(err);
      }
    }, 700);
  });
}

/**
 * 5. POST /api/pipeline/verify -> Real FastAPI Forensic Verification Pipeline
 */
export async function verifyPipeline(
  jobId: string,
  trackIds: number[],
  forceRefresh = false,
  signal?: AbortSignal
): Promise<VerificationResult> {
  const payload: VerifyRequest = {
    job_id: jobId,
    track_ids: trackIds,
    force_refresh: forceRefresh,
  };

  const res = await fetch('/api/pipeline/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  const contentType = res.headers.get('content-type') || '';

  if (!res.ok) {
    let detail = `Verification endpoint failed (${res.status}: ${res.statusText})`;
    if (contentType.includes('application/json')) {
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) {
          detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        // Non-JSON error payload
      }
    }
    throw new Error(detail);
  }

  if (!contentType.includes('application/json')) {
    throw new Error(`Expected JSON from /api/pipeline/verify, but received ${contentType || 'non-JSON'}`);
  }

  return (await res.json()) as VerificationResult;
}

/**
 * 6. Download PDF Report directly as a Blob from pdf_report_url
 */
export async function downloadPdfReport(
  pdfUrl: string,
  downloadFilename: string
): Promise<void> {
  const res = await fetch(pdfUrl);
  if (!res.ok) {
    throw new Error(`Failed to download forensic report from ${pdfUrl} (${res.status} ${res.statusText})`);
  }

  const blob = await res.blob();
  const blobUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = blobUrl;
  anchor.download = downloadFilename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.URL.revokeObjectURL(blobUrl);
}
