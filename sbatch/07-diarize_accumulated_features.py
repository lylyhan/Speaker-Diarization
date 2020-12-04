import os
import sys

sys.path.append("/home/hh2263/denoiser")

#in_dir ='/scratch/hh2263/Spokenweb_data/'
wav_dir = '/scratch/hh2263/Spokenweb_data/enhanced'
#wav_dir = '/scratch/hh2263/Spokenweb_data/clean'
# Define constants.

save_dir = "/scratch/hh2263/Spokenweb_data/enhanced/accumulated"


script_name = os.path.basename(__file__)
script_path = "speakerDiarization_longfiles_ver2.py"


# Create folder.
sbatch_dir = os.path.join(script_name[:-3], "sbatch")
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = os.path.join(script_name[:-3], "slurm")
os.makedirs(slurm_dir, exist_ok=True)


#split work flow
wavDir = []
for file in os.listdir(wav_dir):
    if file.endswith(".wav"):
        file = file.replace(" ","\\ ").replace("(","\\(").replace(")","\\)").replace("&","\\&")
        wavDir.append(file)

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
    
    assigned_wav = wavDir[idx_start:idx_end]
   
    script_path_with_args = []
    for w in assigned_wav:
        script_path_with_args.append(" ".join(
            [script_path, "--save_wavpath="+wav_dir+"/"+w, "--save_pklpath="+save_dir+"/"+w[:-3]+"pkl"]))

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
        f.write("module load gcc/6.3.0\n")
        f.write("\n")
        f.write("cd /home/hh2263/Speaker-Diarization\n")
        #f.write("# The first and second argument is the path to dataset and impulse response files.\n")
        #f.write("# The second and third argument is the range of speaker folders to augment for this job .\n")
        for script in script_path_with_args:
            f.write("python " + script+"\n")


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







