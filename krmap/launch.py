from pathlib    import Path
from time       import sleep
from datetime   import datetime
from itertools  import product
from subprocess import run
from shutil     import rmtree

#cwd         = Path("/home/gonzalo/sw/git/nexus/scan/")
cwd            = Path("/gpfs0/arazi/users/gonzalom/sw/nexus/krmap/")
LT_path        = cwd / "LT"

output         = cwd / "output"
file_job       = cwd / "job.sh"
file_init      = cwd / "init.mac"
file_config    = cwd / "config.mac"
file_detsim    = cwd / "detsim.conf"
file_diomira   = cwd / "diomira.conf"
file_irene     = cwd / "irene.conf"
file_dorothea  = cwd / "dorothea.conf"
file_sophronia = cwd / "sophronia.conf"
file_eutropia  = cwd / "eutropia.conf"

assert cwd           .exists()
assert output        .exists()
assert file_job      .exists()
assert file_init     .exists()
assert file_config   .exists()
assert file_detsim   .exists()
assert file_diomira  .exists()
assert file_irene    .exists()
assert file_dorothea .exists()
assert file_sophronia.exists()
assert file_eutropia .exists()

for subfolder in "conf logs jobs data".split():
    subfolder = output / subfolder
    if subfolder.exists() and subfolder.glob("*"):
        rmtree(subfolder)
    subfolder.mkdir(exist_ok=True)

for f in "conf data".split():
    for subfolder in "nexus detsim diomira irene dorothea sophronia eutropia".split():
        subfolder = output / f / subfolder
        if subfolder.exists() and subfolder.glob("*"):
            rmtree(subfolder)
        subfolder.mkdir(exist_ok=True)

template_job       = open(file_job      ).read()
template_init      = open(file_init     ).read()
template_config    = open(file_config   ).read()
template_detsim    = open(file_detsim   ).read()
template_diomira   = open(file_diomira  ).read()
template_irene     = open(file_irene    ).read()
template_dorothea  = open(file_dorothea ).read()
template_sophronia = open(file_sophronia).read()
template_eutropia  = open(file_eutropia ).read()

seed = 12345678
date = f"{datetime.now()}".replace(" ", "_").replace(":", ".")

def get_seed(fileno):
    return seed + fileno

def log(msg, **kwargs):
    print(msg, **kwargs)
    open(f"launch.{date}.log", "a").write(msg + "\n")


n_events = 2000
n_files  = 5000
for fileno in range(n_files):
    seed     = get_seed(fileno)
    start_id = fileno * n_events

    basename = "kr_{fileno:04}.{ext}"
    out           = output / "logs"               / basename.format(ext="out"   , **globals())
    err           = output / "logs"               / basename.format(ext="err"   , **globals())
    job           = output / "jobs"               / basename.format(ext="sh"    , **globals())
    ini           = output / "conf" / "nexus"     / basename.format(ext="init"  , **globals())
    cnf           = output / "conf" / "nexus"     / basename.format(ext="config", **globals())
    cnf_detsim    = output / "conf" / "detsim"    / basename.format(ext="conf"  , **globals())
    cnf_diomira   = output / "conf" / "diomira"   / basename.format(ext="conf"  , **globals())
    cnf_irene     = output / "conf" / "irene"     / basename.format(ext="conf"  , **globals())
    cnf_dorothea  = output / "conf" / "dorothea"  / basename.format(ext="conf"  , **globals())
    cnf_sophronia = output / "conf" / "sophronia" / basename.format(ext="conf"  , **globals())
    cnf_eutropia  = output / "conf" / "eutropia"  / basename.format(ext="conf"  , **globals())
    out_nexus     = output / "data" / "nexus"     / basename.format(ext="nexus" , **globals())
    out_detsim    = output / "data" / "detsim"    / basename.format(ext="twf"   , **globals())
    out_diomira   = output / "data" / "diomira"   / basename.format(ext="rwf"   , **globals())
    out_irene     = output / "data" / "irene"     / basename.format(ext="pmap"  , **globals())
    out_dorothea  = output / "data" / "dorothea"  / basename.format(ext="kdst"  , **globals())
    out_sophronia = output / "data" / "sophronia" / basename.format(ext="hdst"  , **globals())
    out_eutropia  = output / "data" / "eutropia"  / basename.format(ext="psf"   , **globals())

    open(job          , "w").write(template_job      .format(**globals()))
    open(ini          , "w").write(template_init     .format(**globals()))
    open(cnf          , "w").write(template_config   .format(**globals()))
    open(cnf_detsim   , "w").write(template_detsim   .format(**globals()))
    open(cnf_diomira  , "w").write(template_diomira  .format(**globals()))
    open(cnf_irene    , "w").write(template_irene    .format(**globals()))
    open(cnf_dorothea , "w").write(template_dorothea .format(**globals()))
    open(cnf_sophronia, "w").write(template_sophronia.format(**globals()))
    open(cnf_eutropia , "w").write(template_eutropia .format(**globals()))

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
