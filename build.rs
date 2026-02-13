fn main() {
    let cuda_enabled = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    if !cuda_enabled {
        return;
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        panic!("feature \"cuda\" is supported only on Linux targets");
    }

    cc::Build::new()
        .cuda(true)
        .file("cuda/cuda_matcher.cu")
        .flag("-O3")
        .flag("-lineinfo")
        .compile("cuda_matcher");

    println!("cargo:rerun-if-changed=cuda/cuda_matcher.cu");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
