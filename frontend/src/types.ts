export interface ClipItem {
  id: string;
  filename: string;
  size_bytes: number;
  video_url: string;
  width?: number;
  height?: number;
  duration_sec?: number;
}

export interface LocalClipsResponse {
  directory: string;
  clips: ClipItem[];
}

export type StepType = 'intake' | 'target' | 'verify';

export interface Stage1Track {
  track_id: number;
  vehicle_class: string;
  category: string;
  crop_url: string;
  start_sec: number;
  end_sec: number;
  frame_count: number;
}

export interface FrameBox {
  track_id: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

export interface BoxesData {
  fps: number;
  width: number;
  height: number;
  total_frames?: number;
  frames: Record<string, FrameBox[]>;
}

export interface Stage1Result {
  job_id: string;
  filename: string;
  video_url: string;
  tracked_video_url?: string;
  raw_video_url?: string;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  boxes_url?: string;
  tracks: Stage1Track[];
}

export type JobStatus = 'queued' | 'running' | 'processing' | 'ready' | 'error';

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  stage_step?: number;
  progress?: number;
  message?: string;
  result?: Stage1Result | null;
  error?: string | null;
}

export interface GhostStackItem {
  char: string;
  soft_score: number;
  kind: 'ghost_fill' | 'conflict_alt';
  sources: string[];
}

export interface ReconciledSlot {
  slot_idx: number;
  expected_type: string;
  final_char: string;
  final_confidence: number;
  action: string;
  ui_tile_tone: 'green' | 'black' | 'unknown';
  pipe_a_char?: string | null;
  pipe_b_char?: string | null;
  vlm_char?: string | null;
  ghost_stack: GhostStackItem[];
}

export interface GhostPrediction {
  rank: number;
  plate: string;
  soft_score: number;
  state_code_valid: boolean;
  ghost_slot_indices: number[];
}

export interface VehicleAttributes {
  body_type: string;
  body_type_conf: number;
  colour: string;
  colour_conf: number;
  make: string;
  make_conf: number;
  model: string;
  model_conf: number;
  provider_used: string;
  selected_crops: string[];
}

export interface RegistryMatch {
  rank: number;
  plate_number: string;
  total_confidence: number;
  plate_score: number;
  visual_score: number;
  body_type: string;
  colour: string;
  make: string;
  model: string;
  owner_name: string;
  registered_rto: string;
  registration_year: number;
  sightings?: { camera_id: string; timestamp: string; location?: string }[];
}

export interface EvidenceUrls {
  best_vehicle_crop_url?: string | null;
  vehicle_contact_strip_url?: string | null;
  clearest_plate_crop_url?: string | null;
  plate_grid_unenhanced_url?: string | null;
  plate_grid_enhanced_url?: string | null;
}

export interface VerificationResult {
  case_id: string;
  job_id: string;
  track_key: string;
  filename: string;
  target_id: number;
  merged_track_ids: number[];
  is_merged_track: boolean;
  generated_at: string;
  processing_time_sec: number;
  plate: {
    final_plate: string;
    overall_confidence: number;
    valid_state_code: boolean;
    provider_used: string;
    reconciled_slots: ReconciledSlot[];
    ghost_predictions: GhostPrediction[];
  };
  vehicle: VehicleAttributes;
  registry_matches: RegistryMatch[];
  evidence_urls: EvidenceUrls;
  pdf_report_url: string;
}

export interface VerifyRequest {
  job_id: string;
  track_ids: number[];
  force_refresh?: boolean;
}
