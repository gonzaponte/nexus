{ self
, nixpkgs # <---- This `nixpkgs` has systems removed e.g. legacyPackages.zlib
, ...
}: let

  pkgs = nixpkgs.legacyPackages.extend (import ./add-debug-symbols-to-geant4.nix);

  # Should be able to remove this, once https://github.com/NixOS/nixpkgs/issues/234710 is merged
  clang_16 = (import ./clang_16.nix) pkgs;

  g4 = pkgs.geant4.override {
    enableMultiThreading = false;
    enableInventor       = false;
    enableQt             = true;
    enableXM             = false;
    enableOpenGLX11      = false;
    enablePython         = false;
    enableRaytracerX11   = false;
  };

  g4-data-pkgs = with g4.data; [
    G4PhotonEvaporation
    G4RealSurface
    G4EMLOW
    G4RadioactiveDecay
    G4ENSDFSTATE
    G4SAIDDATA
    G4PARTICLEXS
    G4NDL
  ];

  python-pkgs = pypkgs : with pypkgs; [
    ipython
    pandas
    numpy
    tables
    pytest
    flaky
    hypothesis
    pytest-xdist
    pytest-instafail
    pytest-order
  ];

  other-deps = with pkgs; [
    cmake
    just
    (pkgs.python3.withPackages (python-pkgs))
    gsl
    hdf5
    hdf5.dev
    clang_16
    qt5.wrapQtAppsHook
  ];

  all-deps = [g4] ++ g4-data-pkgs ++ other-deps;

  in {

    devShell = self.devShells.clang;

    devShells.clang = pkgs.mkShell.override { stdenv = clang_16.stdenv; } ( {
      name     = "nexus-clang-devenv";
      packages = all-deps;
      deps     = all-deps;
    });

    # Leading underscore prevents nosys from regenerating this for every system
    _contains-systems = { systems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ]; };
  }
