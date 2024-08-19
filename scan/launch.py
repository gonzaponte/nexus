from pathlib    import Path
from time       import sleep
from datetime   import datetime
from itertools  import product
from subprocess import run
from shutil     import rmtree

#cwd         = Path("/home/gonzalo/sw/git/nexus/scan/")
cwd         = Path("/gpfs0/arazi/users/gonzalom/sw/nexus/scan/")
output      = cwd / "output"
file_init   = cwd / "template.init.mac"
file_config = cwd / "template.config.mac"
file_job    = cwd / "job.sh"

assert cwd        .exists()
assert output     .exists()
assert file_init  .exists()
assert file_config.exists()
assert file_job   .exists()

for subfolder in "conf logs jobs data".split():
    subfolder = output / subfolder
    if subfolder.exists() and subfolder.glob("*"):
        ans = ""
        while ans not in list("yn"):
            ans = input(f"delete existing files in {subfolder} (y/n)? ")
        if ans == "y":
            rmtree(subfolder)
#    continue
    subfolder.mkdir(exist_ok=True)
#break

template_init   = open(file_init  ).read()
template_config = open(file_config).read()
template_job    = open(file_job   ).read()

# Geometry parameters
pitches = [5, 10, 15.6]
el_gaps = [1, 10]
ds_fiber_holder = [0, 2, 5]
ds_anode_holder = [2.5, 5, 10]
params = [pitches, el_gaps, ds_fiber_holder, ds_anode_holder]

batch     = 1
n_photons = 925000
n_sipms   = 35
seeds     = 12345678 - 1
date      = f"{datetime.now()}".replace(" ", "_").replace(":", ".")

def get_seed(pitch, elgap, dfh, dah, fileno):
    pitch = 1 + pitches.index(pitch)
    elgap = 1 + el_gaps.index(elgap)
    dfh   = 1 + ds_fiber_holder.index(dfh)
    dah   = 1 + ds_anode_holder.index(dah)
    s     = f"{pitch:01}{elgap:01}{dfh:01}{dah:01}{fileno:04}"
    return int(s)

def log(msg, **kwargs):
    print(msg, **kwargs)
    open(f"launch.{date}.log", "a").write(msg + "\n")

for pitch, el_gap, d_fiber_holder, d_anode_holder in product(*params):
    n_events = 1000 if pitch > 5 else 100
    n_files  =   10 if pitch > 5 else 100
    for fileno in range(n_files):
        fileno += (batch-1)*n_files

        seed = get_seed(pitch, el_gap, d_fiber_holder, d_anode_holder, fileno)

        basename = "p_{pitch:.1f}_elgap_{el_gap}_dfh_{d_fiber_holder}_dah_{d_anode_holder:.1f}_subfile_{fileno:04}.{ext}"
        ini = output / "conf" / basename.format(ext="init"  , **globals())
        cnf = output / "conf" / basename.format(ext="config", **globals())
        out = output / "logs" / basename.format(ext="out"   , **globals())
        err = output / "logs" / basename.format(ext="err"   , **globals())
        job = output / "jobs" / basename.format(ext="sh"    , **globals())
        txt = output / "data" / basename.format(ext="txt"   , **globals())

        open(ini, "w").write(template_init  .format(**globals()))
        open(cnf, "w").write(template_config.format(**globals()))
        open(job, "w").write(template_job   .format(**globals()))

        while True:
            text = run("qstat -q arazi.q".split(), capture_output=True).stdout
            njob = sum(1 for _ in filter(lambda line: "gonzalom" in line, text.decode().split("\n")))
            if njob < 180: break
            log(f"{datetime.now()} | waiting to submit job {job}...")
            sleep(60)

        command = f"qsub -q arazi.q -o {out} -e {err} {job}"

        while True:
            log(f"{datetime.now()} | submitting job {job}...")
            result = run(command.split(), capture_output=True)
            result = "\n".join([result.stdout.decode(), result.stderr.decode()])
            log(result)
            if "error" not in result: break
            sleep(60)
