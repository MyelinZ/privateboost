{ pkgs, ... }: {
  languages.rust = {
    enable = true;
    channel = "stable";
    components = [
      "rustc"
      "cargo"
      "clippy"
      "rustfmt"
      "rust-analyzer"
      # cargo-llvm-cov needs the toolchain's own llvm-cov/llvm-profdata
      # (the profraw format must match rustc's LLVM version).
      "llvm-tools"
    ];
  };
  # Python + uv drive the experiment scripts (scripts/, pyproject.toml);
  # texliveFull builds manuscript/main.tex so a fresh clone reproduces
  # results, figures, and the paper itself without host packages.
  languages.python = {
    enable = true;
    uv.enable = true;
  };
  # manylinux wheels in the uv venv (numpy, scipy, xgboost) dlopen these from
  # the environment; without them imports fail with "libz.so.1 ... not found"
  # depending on interpreter import order.
  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.zlib pkgs.stdenv.cc.cc.lib ];
  packages = [
    pkgs.protobuf
    pkgs.just
    pkgs.openssl
    pkgs.texliveFull
    pkgs.cargo-nextest
    pkgs.cargo-llvm-cov
    pkgs.gitleaks
  ];
  env.PROTOC = "${pkgs.protobuf}/bin/protoc";
  # Loads the git-ignored .env (HCLOUD_TOKEN for deploy/hetzner/infra).
  dotenv.enable = true;
}
