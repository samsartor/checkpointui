use std::{env, path::PathBuf};

pub fn main() {
    use cmake;
    use cmake::Config;

    let dst = Config::new("ggml")
        .build_target("ggml-base")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();
    let lib_path = dst.join("build").join("src");

    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=static=ggml-base");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
