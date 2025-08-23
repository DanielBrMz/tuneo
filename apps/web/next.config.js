/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation.
 * This is especially useful for Docker builds.
 */
await import("./src/env.js");

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: true,
  swcMinify: true,
  
  /** 
   * Enable standalone output for Docker deployment
   */
  output: 'standalone',

  /**
   * Configure Rust Audio Engine URL
   */
  env: {
    RUST_AUDIO_ENGINE_URL: process.env.RUST_AUDIO_ENGINE_URL || 'http://localhost:8080',
  },
};

export default config;