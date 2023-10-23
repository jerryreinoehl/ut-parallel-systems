import subprocess
import re


def hash_time(hash_workers=1, input="coarse"):
    runs = 20
    total_time = 0
    time_re = re.compile("hashTime: (.*)")

    cmd = [
        "./bst",
        "-input",
        f"data/{input}.txt",
        f"-hash-workers={hash_workers}",
    ]
    print(" ".join(cmd))

    for i in range(runs):
        output = subprocess.check_output(cmd).decode()
        match = time_re.match(output)
        if time_re.match(output):
            total_time += float(match[1])

    avg_time = total_time / runs
    print(f"avg_time for {runs} runs is {avg_time * 1000:0.4f} ms")


def hash_group_time(hash_workers=1, data_workers=1, mutex=False, input="coarse"):
    runs = 20
    total_time = 0
    time_re = re.compile("hashGroupTime: (.*)")

    cmd = [
        "./bst",
        "-input",
        f"data/{input}.txt",
        f"-hash-workers={hash_workers}",
        f"-data-workers={data_workers}",
    ]
    if mutex:
        cmd.append("-add-with-mutex")
    print(" ".join(cmd))

    for i in range(runs):
        output = subprocess.check_output(cmd).decode()
        match = time_re.match(output)
        if time_re.match(output):
            total_time += float(match[1])

    avg_time = total_time / runs
    print(f"avg_time for {runs} runs is {avg_time * 1000:0.4f} ms")


def compare_tree_time(
    hash_workers=16, data_workers=1, comp_workers=1, mutex=False, input="coarse",
    buffer=False
):
    runs = 20
    total_time = 0
    time_re = re.compile("compareTreeTime: (.*)")

    cmd = [
        "./bst",
        "-input",
        f"data/{input}.txt",
        f"-hash-workers={hash_workers}",
        f"-data-workers={data_workers}",
        f"-comp-workers={comp_workers}",
    ]
    if mutex:
        cmd.append("-add-with-mutex")

    if buffer:
        cmd.append("-compare-with-buffer")

    print(" ".join(cmd))

    for i in range(runs):
        output = subprocess.check_output(cmd).decode()
        match = time_re.search(output)
        if match:
            total_time += float(match[1])

    avg_time = total_time / runs
    print(f"avg_time for {runs} runs is {avg_time * 1000:0.4f} ms")


def main():
    #hash_time(hash_workers=75000, input="fine")
    #hash_time(hash_workers=64, input="coarse")
    #hash_group_time(hash_workers=1, data_workers=32, mutex=False, input="fine")
    compare_tree_time(comp_workers=2, buffer=False, input="fine")


if __name__ == "__main__":
    main()
