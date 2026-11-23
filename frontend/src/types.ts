// TypeScript types for Invictus Vehicle Trace Surveillance Dashboard (AIML-01)

export type BodyType = 'car' | 'suv' | 'two-wheeler' | 'auto-rickshaw' | 'truck' | 'bus';

export interface TrackSummary {
  track_id: number;
  coarse_class: string;
  first_sec: number;
  last_sec: number;
  frames_seen: number;
  thumbnail_url?: string | null;
}

export interface VideoScanResponse {
  video_id?: string | null;
  video_path?: string | null;
  duration_seconds: number;
  total_frames: number;
  fps: number;
  tracks_count: number;
  tracks: TrackSummary[];
}

export interface FrameEvidence {
  frame_idx: number;
  timestamp_sec?: number | null;
  image_path: string;
  plate_crop_path?: string | null;
  ocr_text?: string | null;
  confidence: number;
}

export interface VehicleEvidence {
  track_id: number;
  consensus_plate: string;
  char_confidences: number[];
  colour: string;
  colour_conf: number;
  body_type: BodyType;
  body_type_conf: number;
  make?: string;
  make_conf?: number;
  model?: string;
  model_conf?: number;
  frames_used: FrameEvidence[];
  warning?: string | null;
}

export interface ScoreBreakdown {
  plate_score: number;
  colour_score: number;
  body_type_score: number;
  make_model_score: number;
}

export interface MatchCandidate {
  rank: number;
  plate_number: string;
  body_type: string;
  colour: string;
  make: string;
  model: string;
  rto_city: string;
  synthetic_owner_id: string;
  composite_score: number;
  score_breakdown: ScoreBreakdown;
  match_reasons: string[];
}

export interface AnalysisResponse {
  track_id: number;
  evidence: VehicleEvidence;
  top_matches: MatchCandidate[];
  no_confident_match: boolean;
  explanation?: string | null;
  warning?: string | null;
}

export interface DemoScenario {
  id: string;
  title: string;
  description: string;
  evidence: VehicleEvidence;
  suggested_known_partial_plate?: string | null;
  expected_outcome: string;
}

export interface SystemHealth {
  status: string;
  service: string;
  version: string;
  database_connected: boolean;
  registry_vehicles_count: number;
}
