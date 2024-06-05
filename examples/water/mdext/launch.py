import os

for seed in range(0, 640):
    print(f"Seed {seed}: ", end="", flush=True)
    runPath = f"Runs/seed{seed:04}"
    os.system(f"mkdir -p {runPath}")
    os.chdir(runPath)
    os.system(f"sbatch ../../run.job {seed}")
    os.chdir("../..")
