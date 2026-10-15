#!/usr/bin/env bash
set -u
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
py=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
out=artifacts/phase21/candidate_gate_generation_resume_20261014_v1
CUDA_VISIBLE_DEVICES=GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc timeout --kill-after=30 21900 "$py" -u -m experiment.phase21.resume_candidate_gate_generation worker Toys > "$out/Toys/worker.log" 2>&1 &
toys_pid=$!
CUDA_VISIBLE_DEVICES=GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253 timeout --kill-after=30 21900 "$py" -u -m experiment.phase21.resume_candidate_gate_generation worker Beauty > "$out/Beauty/worker.log" 2>&1 &
beauty_pid=$!
wait "$toys_pid"
toys_exit=$?
wait "$beauty_pid"
beauty_exit=$?
printf 'Toys exit=%s; Beauty exit=%s\n' "$toys_exit" "$beauty_exit" > "$out/worker_exits.txt"
if [ "$toys_exit" = 0 ] && [ "$beauty_exit" = 0 ]; then
    "$py" -u -m experiment.phase21.resume_candidate_gate_generation summarize > "$out/finalizer.log" 2>&1
else
    exit 1
fi
