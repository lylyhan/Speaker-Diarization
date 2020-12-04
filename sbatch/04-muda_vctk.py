import os
import sys

sys.path.append("../")

SRC_path='/scratch/hh2263/VCTK/VCTK-Corpus/wav48'
ir_path = '/home/hh2263/Speaker-Diarization/ir_files/'
bg_path = '/home/hh2263/Speaker-Diarization/bg_files/'
# Define constants.






script_name = os.path.basename(__file__)
script_path = "/home/hh2263/Speaker-Diarization/augmentation_ver3.py"


# Create folder.
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)


#split work flow
wavDir = os.listdir(SRC_path)
num_spk = len(wavDir)
num_jobs = 19
avg_spk = round(num_spk/num_jobs)
#assign each job which speaker to augment
for i in range(num_jobs):
    idx_start= i*avg_spk
    if i == num_jobs-1:
        idx_end = num_spk
    else:
        idx_end = (i+1)*avg_spk
    

    script_path_with_args = " ".join(
        [script_path, SRC_path, ir_path, str(idx_start), str(idx_end), bg_path])
    
    job_name = "_".join([
        script_name[:2],
        "start-" + str(idx_start), "end" +  str(idx_end)])

    file_name = job_name + ".sbatch"
    file_path = os.path.join(sbatch_dir, file_name)

    # Generate file.
    with open(file_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("#BATCH --job-name=" + script_name[:2] + "\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --tasks-per-node=1\n")
        f.write("#SBATCH --cpus-per-task=10\n")
        f.write("#SBATCH --time=15:00:00\n")
        f.write("#SBATCH --mem=64GB\n")
        f.write("#SBATCH --output=" +\
            "../slurm/" + job_name + "_%j.out\n")
        f.write("\n")
        f.write("module purge\n")
        f.write("\n")
        f.write("# The first and second argument is the path to dataset and impulse response files.\n")
        f.write("# The second and third argument is the range of speaker folders to augment for this job .\n")
        f.write("python " + script_path_with_args)


# Open shell file.
file_path = os.path.join(sbatch_dir, script_name[:2] + ".sh")

with open(file_path, "w") as f:
    # Print header.
    f.write("# This shell script augment VCTK dataset using muda in a parallized way.\n")
    f.write("\n")

    # Loop over folds: training and validation.
    for i in range(num_jobs):
        idx_start= i*avg_spk
        if i == num_jobs-1:
            idx_end = num_spk
        else:
            idx_end = (i+1)*avg_spk
       
        # Define job name.
        job_name = "_".join([
        script_name[:2],
        "start-" + str(idx_start), "end" +  str(idx_end)])

        sbatch_str = "sbatch " + job_name + ".sbatch"

        # Write SBATCH command to shell file.
        f.write(sbatch_str + "\n")
        f.write("\n")

# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)
