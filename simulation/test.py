import subprocess
import os

EXE_PATH = "C:\\Users\\creiner\\Agent-Based-Simulation-Model\\Agent-Based-Simulation\\bin"
args = [
    "-j",
    "-n", "10",
    "-p", "5",
    "-h", "8",
    "-w", "5",
    "-o", "3",
    "-m", "1",
    "-r", "2",
    "-d", "3",
]

cmd = ["cmd", "/c", EXE_PATH, *args]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

out, err = proc.communicate()

print("RETURN CODE:", proc.returncode)
print("STDOUT:")
print(repr(out))
print(out)
print("STDERR:")
print(repr(err))
print(err)