#!/usr/bin/env python3

import shutil
import subprocess

template_init = open("macros/meshes_template.init.mac").read()
template_conf = open("macros/meshes_template.config.mac").read()

refls = [0 + 2.5*i for i in range(15)]

for refl in refls:
    tag     = f"{refl:.1f}"
    init    = f"/tmp/mesh_{tag}.init"
    conf    = f"/tmp/mesh_{tag}.conf"
    outfile = f"mesh_reflectivity_{tag}"
    open(init, "w").write(template_init.format(**globals()))
    open(conf, "w").write(template_conf.format(**globals()))

    n = 10**6
    command = f"just run {init} -n {n}"
    print("Executing command:", command)
    subprocess.run(command.split(), capture_output=True)
