/**
 * Shared types between frontend and Rust backend
 */

export interface AudioFingerprint {
  id: string;
  hash: string;
  duration: number;
  sampleRate: number;
  createdAt: Date;
}

export interface TrackMetadata {
  title: string;
  artist: string;
  album?: string;
  year?: number;
  genre?: string;
}

export interface RecognitionResult {
  matched: boolean;
  confidence: number;
  track?: TrackMetadata;
  fingerprint: AudioFingerprint;
}