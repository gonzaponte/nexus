#!/usr/bin/env python3

#!/usr/bin/env python3

import shutil
import subprocess

template_init = open("macros/NEXT100_fullKr_template.init.mac").read()
template_conf = open("macros/NEXT100_fullKr_template.config.mac").read()

processes = []
for i in range(8):
    tag     = str(i)
    init    = f"/tmp/kr_{tag}.init"
    conf    = f"/tmp/kr_{tag}.conf"
    outfile = f"kr_{tag}.nexus"
    open(init, "w").write(template_init.format(**globals()))
    open(conf, "w").write(template_conf.format(**globals()))

    n = 125
    command = f"just run {init} -n {n}"
    print("Executing command:", command)
    p = subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(p)

for i, p in enumerate(processes):
    print(f"Waiting for process {i}")
    p.wait()
