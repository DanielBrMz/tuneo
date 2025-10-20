//! Tuneo Native Core
//! Shared library for all native applications (iOS, Android, Desktop)

pub mod audio;
pub mod api;
pub mod cache;
pub mod error;

// Re-exports
pub use error::{Result, TuneoError};

// Rust API (for Rust consumers like Linux desktop)
pub fn init() -> bool {
    #[cfg(target_os = "android")]
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug),
    );
    
    #[cfg(not(target_os = "android"))]
    let _ = env_logger::try_init();
    
    log::info!("Tuneo Native Core initialized");
    true
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// C FFI exports (for iOS/Android/Windows)
#[cfg(feature = "ffi")]
pub mod ffi {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    #[no_mangle]
    pub extern "C" fn tuneo_init() -> bool {
        init()
    }

    #[no_mangle]
    pub extern "C" fn tuneo_version() -> *mut c_char {
        let version = version();
        match CString::new(version) {
            Ok(s) => s.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[no_mangle]
    pub extern "C" fn tuneo_free_string(s: *mut c_char) {
        if !s.is_null() {
            unsafe {
                drop(CString::from_raw(s));
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn tuneo_fingerprint_audio(path: *const c_char) -> *mut c_char {
        let path_str = unsafe {
            if path.is_null() {
                return std::ptr::null_mut();
            }
            match CStr::from_ptr(path).to_str() {
                Ok(s) => s,
                Err(_) => return std::ptr::null_mut(),
            }
        };
        
        log::info!("Fingerprinting audio: {}", path_str);
        
        let result = serde_json::json!({
            "success": true,
            "fingerprint": format!("fp_{}", path_str.len()),
            "duration_ms": 0,
            "sample_rate": 44100
        });
        
        match CString::new(result.to_string()) {
            Ok(s) => s.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[no_mangle]
    pub extern "C" fn tuneo_recognize(fingerprint: *const c_char) -> *mut c_char {
        let fp_str = unsafe {
            if fingerprint.is_null() {
                return std::ptr::null_mut();
            }
            match CStr::from_ptr(fingerprint).to_str() {
                Ok(s) => s,
                Err(_) => return std::ptr::null_mut(),
            }
        };
        
        log::info!("Recognizing fingerprint: {}", fp_str);
        
        let result = serde_json::json!({
            "matched": false,
            "confidence": 0.0,
            "track_name": null,
            "artist": null
        });
        
        match CString::new(result.to_string()) {
            Ok(s) => s.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        assert!(init());
    }

    #[test]
    fn test_version() {
        let version = version();
        assert!(!version.is_empty());
    }
}
