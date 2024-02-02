{ self
, nixpkgs # <---- This `nixpkgs` has systems removed e.g. legacyPackages.zlib
, ...
}: let

  pkgs = nixpkgs.legacyPackages.extend (import ./add-debug-symbols-to-geant4.nix);

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

  # Should be able to remove this, once https://github.com/NixOS/nixpkgs/issues/234710 is merged
  clang_16 = (import ./clang_16.nix) pkgs;

  shell-shared = {
      G4_DIR = "${pkgs.geant4}";
      QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.libsForQt5.qt5.qtbase.bin}/lib/qt-${pkgs.libsForQt5.qt5.qtbase.version}/plugins";

      shellHook = ''
          export NEXUS_LIB=$PWD/build/
          export LD_LIBRARY_PATH=$NEXUS_LIB/lib:$LD_LIBRARY_PATH;
          export CMAKE_PREFIX_PATH=$NEXUS_LIB/lib/cmake/:$CMAKE_PREFIX_PATH;

          # TODO replace manual envvar setting with with use of packages' setupHooks
          export G4NEUTRONHPDATA="${g4.data.G4NDL}/share/Geant4-11.0.4/data/G4NDL4.6"
          export G4L EDATA="${g4.data.G4EMLOW}/share/Geant4-11.0.4/data/G4EMLOW8.0"
          export G4LEVELGAMMADATA="${g4.data.G4PhotonEvaporation}/share/Geant4-11.0.4/data/G4PhotonEvaporation5.7"
          export G4RADIOACTIVEDATA="${g4.data.G4RadioactiveDecay}/share/Geant4-11.0.4/data/G4RadioactiveDecay5.6"
          export G4PARTICLEXSDATA="${g4.data.G4PARTICLEXS}/share/Geant4-11.0.4/data/G4PARTICLEXS4.0"
          export G4PIIDATA="${g4.data.G4PII}/share/Geant4-11.0.4/data/G4PII1.3"
          export G4REALSURFACEDATA="${g4.data.G4RealSurface}/share/Geant4-11.0.4/data/G4RealSurface2.2"
          export G4SAIDXSDATA="${g4.data.G4SAIDDATA}/share/Geant4-11.0.4/data/G4SAIDDATA2.0"
          export G4ABLADATA="${g4.data.G4ABLA}/share/Geant4-11.0.4/data/G4ABLA3.1"
          export G4INCLDATA="${g4.data.G4INCL}/share/Geant4-11.0.4/data/G4INCL1.0"
          export G4ENSDFSTATEDATA="${g4.data.G4ENSDFSTATE}/share/Geant4-11.0.4/data/G4ENSDFSTATE2.3"
      '';
  };

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

    devShells.clang = pkgs.mkShell.override { stdenv = clang_16.stdenv; } (shell-shared // {
      name     = "nexus-clang-devenv";
      packages = all-deps;
      deps     = all-deps;

      HDF5_DIR = pkgs.symlinkJoin { name = "hdf5"; paths = [ pkgs.hdf5 pkgs.hdf5.dev ]; };
#      HDF5_LIB = "${HDF5_DIR}/lib";
#      HDF5_INC = "${HDF5_DIR}/include";
    });

    # Leading underscore prevents nosys from regenerating this for every system
    _contains-systems = { systems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ]; };
  }
