#pragma once

/**
 * Initialize the native core
 * Returns true if initialization was successful
 */
bool tuneo_init(void);

/**
 * Get version string
 */
char *tuneo_version(void);

/**
 * Free a string allocated by Rust
 */
void tuneo_free_string(char *s);

/**
 * Fingerprint audio from file path
 * Returns JSON string with fingerprint data
 */
char *tuneo_fingerprint_audio(const char *path);

/**
 * Recognize audio from fingerprint
 * Returns JSON string with recognition result
 */
char *tuneo_recognize(const char *fingerprint);
