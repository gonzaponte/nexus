
# Needed to make `"$@"` usable in recipes
set positional-arguments := true

branch     := `git branch --show-current`
build-path := "builds/" + branch
ncores     := `nproc`

build j=ncores:
  mkdir -p {{build-path}}
  cmake -S . -B {{build-path}}
  cmake --build {{build-path}} -- -j {{j}}

run *ARGS:
  {{build-path}}/nexus -b "$@"

view *ARGS:
  sh execute-with-nixgl-if-needed.sh {{build-path}}/nexus -i "$@"

build-and-run *ARGS: build
  just run "$@"

clean:
  rm {{build-path}} -rf
