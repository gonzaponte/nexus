from pathlib    import Path
from time       import sleep
from datetime   import datetime
from itertools  import product
from subprocess import run
from shutil     import rmtree

#cwd         = Path("/home/gonzalo/sw/git/nexus/refl_length/")
cwd         = Path("/gpfs0/arazi/users/gonzalom/sw/nexus/refl_length/")
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

reflectivities = [0.95+0.01*i for i in range(5)]
lengths = [5*(i+1) for i in range(20)] # mm

n_photons = 10**7
seed      = 12345678 - 1
date      = f"{datetime.now()}".replace(" ", "_").replace(":", ".")

def log(msg, **kwargs):
    print(msg, **kwargs)
    open(f"launch.{date}.log", "a").write(msg + "\n")

for reflectivity, length in product(reflectivities, lengths):
    seed += 1

    basename = "reflectivity_{reflectivity}_length_{length:03}{tag}.{ext}"
    ini  = output / "conf" / basename.format(ext="init"  , tag=""     , **globals())
    cnf  = output / "conf" / basename.format(ext="config", tag=""     , **globals())
    out  = output / "logs" / basename.format(ext="out"   , tag=""     , **globals())
    err  = output / "logs" / basename.format(ext="err"   , tag=""     , **globals())
    job  = output / "jobs" / basename.format(ext="sh"    , tag=""     , **globals())
    tpb  = output / "data" / basename.format(ext="txt"   , tag="_tpb" , **globals())
    sipm = output / "data" / basename.format(ext="txt"   , tag="_sipm", **globals())

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
