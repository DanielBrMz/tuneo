/**
 * Placeholder for shared router logic
 * This can be expanded to share tRPC routers across multiple apps
 */

export const AUDIO_ENGINE_ENDPOINTS = {
  fingerprint: '/api/fingerprint',
  recognize: '/api/recognize',
  stream: '/api/stream',
} as const;