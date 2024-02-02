final: previous: {
  geant4 = previous.geant4.overrideAttrs (old: {
    dontStrip = true;
    NIX_CFLAGS_COMPILE =
      (if builtins.hasAttr "NIX_CFLAGS_COMPILE" old then old.NIX_CFLAGS_COMPILE else "")
      + " -ggdb -Wa,--compress-debug-sections";
    });
}
