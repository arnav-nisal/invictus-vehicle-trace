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
  forceRefresh = false
): Promise<JobStatusResponse> {
  const res = await fetch('/api/videos/select-local', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, force_refresh: forceRefresh }),
  });

  if (!res.ok) {
    let detail = `Server responded with ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Non-JSON error payload
    }
    throw new Error(detail);
  }

  return (await res.json()) as JobStatusResponse;
}

/**
 * 2. GET /api/videos/{job_id}/status
 * Polls Stage 1 execution status from FastAPI backend.
 */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`/api/videos/${encodeURIComponent(jobId)}/status`);

  if (!res.ok) {
    let detail = `Status request failed (${res.status}: ${res.statusText})`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Non-JSON error payload
    }
    throw new Error(detail);
  }

  return (await res.json()) as JobStatusResponse;
}

/**
 * 3. GET boxes from boxes_url directly from backend / outputs.
 */
export async function fetchBoxesData(boxesUrl: string): Promise<BoxesData> {
  const res = await fetch(boxesUrl);
  if (!res.ok) {
    throw new Error(`Failed to load bounding boxes from ${boxesUrl} (${res.status} ${res.statusText})`);
  }

  const data = await res.json();
  if (!data || typeof data !== 'object' || !data.frames) {
    throw new Error(`Invalid bounding boxes payload received from ${boxesUrl}`);
  }

  return data as BoxesData;
}

/**
 * 4. Polling helper: Polls GET /api/videos/{job_id}/status every 700ms
 */
export async function pollStage1Job(
  jobId: string,
  onProgress?: (statusResp: JobStatusResponse) => void
): Promise<Stage1Result> {
  // First immediate status check
  const initial = await getJobStatus(jobId);
  if (onProgress) {
    onProgress(initial);
  }
  if (initial.status === 'ready' && initial.result) {
    return initial.result;
  }
  if (initial.status === 'error') {
    throw new Error(initial.error || 'Stage 1 processing failed on server');
  }

  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const resp = await getJobStatus(jobId);
        if (onProgress) {
          onProgress(resp);
        }

        if (resp.status === 'ready' && resp.result) {
          clearInterval(interval);
          resolve(resp.result);
        } else if (resp.status === 'error') {
          clearInterval(interval);
          reject(new Error(resp.error || 'Stage 1 processing failed on server'));
        }
      } catch (err) {
        clearInterval(interval);
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
  forceRefresh = false
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
  });

  if (!res.ok) {
    let detail = `Verification endpoint failed (${res.status}: ${res.statusText})`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Non-JSON error payload
    }
    throw new Error(detail);
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
