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
  /** False when no plate was detected (visual-only tracking mode). */
  plate_detected?: boolean;
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
  /** RTO spatial prior probability distribution, e.g. {"MH14": 0.80, "MH12": 0.12}. */
  rto_priors?: Record<string, number> | null;
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
  owner_name?: string | null;
  registered_rto?: string | null;
  registration_year?: number | null;
  watchlist_status: 'CLEAR' | 'STOLEN' | 'FLAGGED' | string;
  composite_score: number;
  /** Forensic total confidence (plate + rto + visual fusion). */
  total_confidence: number;
  /** Plate-only score from OCR confusion-aware alignment. */
  plate_score?: number | null;
  /** RTO spatial prior score. */
  rto_prior_score?: number | null;
  /** Visual VLM attribute score. */
  visual_score?: number | null;
  /** Character-by-character alignment diff string from backend. */
  character_diff?: string | null;
  /** Forensic anomaly flags, e.g. ["SUSPECTED_CLONED_PLATE"]. */
  anomaly_flags: string[];
  score_breakdown: ScoreBreakdown;
  match_reasons: string[];
}

export interface SightingRecord {
  sighting_id: number;
  track_id?: number | null;
  camera_id: string;
  timestamp: string;
  top_matched_plate: string;
  confidence: number;
  crop_path?: string | null;
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
