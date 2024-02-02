pkgs: if pkgs.stdenv.isDarwin
      then pkgs.llvmPackages_16.clang.override rec {
        libc = pkgs.darwin.Libsystem;
        bintools = pkgs.bintools.override { inherit libc; };
        inherit (pkgs.llvmPackages) libcxx;
        extraPackages = [
          pkgs.llvmPackages.libcxxabi
          # Use the compiler-rt associated with clang, but use the libc++abi from the stdenv
          # to avoid linking against two different versions (for the same reasons as above).
          (pkgs.llvmPackages_16.compiler-rt.override {
            inherit (pkgs.llvmPackages) libcxxabi;
          })
        ];
      }
      else pkgs.llvmPackages_16.clang
