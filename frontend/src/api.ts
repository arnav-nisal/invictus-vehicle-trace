import {
  AnalysisResponse,
  DemoScenario,
  SystemHealth,
  VehicleEvidence,
  VideoScanResponse,
} from './types';

// Use relative URL by default to leverage Vite's dev-server proxy and prevent CORS issues.
// If VITE_API_BASE is provided in the environment, use it instead.
const API_BASE = (import.meta as any)?.env?.VITE_API_BASE || '';

export function resolveImageUrl(path?: string | null): string {
  if (!path) return '';
  if (path.startsWith('data:')) return path;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return API_BASE ? `${API_BASE}/${cleanPath}` : `/${cleanPath}`;
}

export async function getSystemHealth(): Promise<SystemHealth> {
  const resp = await fetch(`${API_BASE}/api/health`);
  if (!resp.ok) {
    throw new Error(`Health check failed with status ${resp.status}`);
  }
  return resp.json();
}

export async function getDemoScenarios(): Promise<DemoScenario[]> {
  const resp = await fetch(`${API_BASE}/api/demo-scenarios`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch demo scenarios: ${resp.status}`);
  }
  return resp.json();
}

export async function scanVideo(
  file?: File | null,
  clipId?: string
): Promise<VideoScanResponse> {
  let resp: Response;
  if (file) {
    resp = await fetch(`${API_BASE}/api/videos/scan`, {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'video/mp4',
      },
      body: file,
    });
  } else {
    resp = await fetch(`${API_BASE}/api/videos/scan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ clip_id: clipId || 'demo-traffic-1' }),
    });
  }

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`Video scan failed (${resp.status}): ${errorText}`);
  }
  return resp.json();
}

export async function analyzeVehicle(params: {
  scenario_id?: string;
  evidence?: VehicleEvidence;
  known_partial_plate?: string;
}): Promise<AnalysisResponse> {
  const resp = await fetch(`${API_BASE}/api/vehicles/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`Vehicle analysis failed (${resp.status}): ${errorText}`);
  }
  return resp.json();
}

export async function exportReportHtml(analysis: AnalysisResponse): Promise<string> {
  const resp = await fetch(`${API_BASE}/api/reports/export`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(analysis),
  });

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`Report export failed (${resp.status}): ${errorText}`);
  }
  return resp.text();
}
