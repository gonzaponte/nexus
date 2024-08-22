import sys

from pathlib    import Path
from time       import sleep
from datetime   import datetime
from subprocess import run

#cwd          = Path("/home/gonzalo/sw/git/nexus/scan/")
cwd          = Path("/gpfs0/arazi/users/gonzalom/sw/nexus/scan/")
folder       = Path(sys.argv[1])
inputfolder  = folder / "pairs"
outputfolder = folder / "output"
file_job     = cwd    / "decojob.sh"

outputfolder.mkdir(exist_ok=True)

assert cwd         .exists()
assert inputfolder .exists()
assert outputfolder.exists()
assert file_job    .exists()

for subfolder in "logs jobs data".split():
    subfolder = outputfolder / subfolder
    if subfolder.exists() and subfolder.glob("*"):
        ans = ""
        while ans not in list("yn"):
            ans = input(f"delete existing files in {subfolder} (y/n)? ")
        if ans == "y":
            rmtree(subfolder)
    subfolder.mkdir(exist_ok=True)

template_job = open(file_job).read()

date = f"{datetime.now()}".replace(" ", "_").replace(":", ".")

def log(msg, **kwargs):
    print(msg, **kwargs)
    open(f"launch.{date}.log", "a").write(msg + "\n")

for filename in inputfolder.glob("*"):
    out = outputfolder / "logs" / (filename.stem + ".out")
    err = outputfolder / "logs" / (filename.stem + ".err")
    job = outputfolder / "jobs" / (filename.stem + ".sh" )

    open(job, "w").write(template_job.format(**globals()))

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
