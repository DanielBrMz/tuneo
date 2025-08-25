fn main() {
    #[cfg(feature = "ffi")]
    {
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        
        cbindgen::Builder::new()
            .with_crate(crate_dir)
            .with_language(cbindgen::Language::C)
            .with_no_includes()
            .with_pragma_once(true)
            .with_parse_expand(&["tuneo-native-core"])
            .with_parse_expand_features(&["ffi"])
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file("tuneo_native_core.h");
    }
}
