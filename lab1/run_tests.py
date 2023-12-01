#!/usr/bin/env python3
import os
from subprocess import check_output
import re
from time import sleep
import uuid
from pathlib import Path
import random
import subprocess

#
#  Feel free (a.k.a. you have to) to modify this to instrument your code
#

THREADS = [0]
LOOPS = [1, 10]
INPUTS = ["seq_64_test.txt"]


class PrefixScanTest():

    EXE = "./bin/prefix_scan"
    TIME_RE = re.compile(r"time: (\d+)")

    def __init__(
        self, num_elements, num_threads, num_loops=100000, test_dir: Path = None
    ):
        self._num_elements = num_elements
        self._num_threads = num_threads
        self._num_loops = num_loops
        self._test_dir = test_dir or Path("./tests")
        self._filename = None
        self._output_file = "out.txt"
        self._cmd = None
        self._time = None
        self._expected = None
        self._generated = None

    def run(self):
        elements = [random.randint(-10000, 10000) for _ in range(self._num_elements)]
        self._expected = self._prefix_sum(elements)
        self._write_test(elements)

        self._run_prefix_sum()
        self._generated = self._read_test()

        return self._expected == self._generated

    def _prefix_sum(self, elements):
        elements = elements[:]  # Create a copy of `elements`.

        for i in range(1, len(elements)):
            elements[i] += elements[i - 1]

        return elements

    def _write_test(self, elements):
        self._filename = self._test_dir / uuid.uuid4().hex

        with open(self._filename, "w") as f:
            print(len(elements), file=f)
            print("\n".join(str(elem) for elem in elements), file=f)

    def _read_test(self):
        with open(self._output_file, "r") as f:
            elements = [int(line) for line in f]

        return elements

    def _run_prefix_sum(self):
        cmd = [
            self.EXE,
            "-i", str(self._filename),
            "-n", str(self._num_threads),
            "-l", str(self._num_loops),
            "-o", self._output_file,
            "-s",
        ]

        self._cmd = " ".join(cmd)
        out = check_output(cmd).decode()

        match = self.TIME_RE.match(out)
        if match:
            self._time = int(match[1])

    @property
    def cmd(self):
        return self._cmd

    @property
    def time(self):
        return self._time

    @property
    def expected(self):
        return self._expected

    @property
    def generated(self):
        return self._generated


class PrefixScanRunner():

    EXE = "./bin/prefix_scan"
    TIME_RE = re.compile(r"time: (\d+)")

    def __init__(self, input_file, num_threads, num_loops=100000):
        self._input_file = input_file
        self._num_threads = num_threads
        self._num_loops = num_loops
        self._output_file = "out.txt"
        self._time = None

    def run(self):
        cmd = [
            self.EXE,
            "-i", str(self._input_file),
            "-n", str(self._num_threads),
            "-l", str(self._num_loops),
            "-o", self._output_file,
        ]

        self._cmd = " ".join(cmd)
        out = check_output(cmd).decode()

        match = self.TIME_RE.match(out)
        if match:
            self._time = int(match[1])

    @property
    def time(self):
        return self._time


def main():
    for i in range(500):
        elements = random.randint(1, 1000)
        threads = random.randint(1, 12)
        print(elements, threads)
        test = PrefixScanTest(elements, threads)
        passed = test.run()
        print(passed, test.time)
        if passed == False:
            print("\x1b[1;31Failed!!!\x1b[0m")
    exit(0)

    out = open("test4.csv", "w")

    for n_threads in [2, 4, 8, 10, 16]:
        out.write(f"\n{n_threads},")
        for test in ["1k.txt", "8k.txt", "16k.txt", "32k.txt"][1:2]:
            print(f"======== {n_threads}, {test} ========")
            runner = PrefixScanRunner(f"tests/{test}", n_threads, num_loops=1000000)
            total = 0
            n_warmup_runs = 3
            n_runs = 10
            avg = 0

            print("======== Warmup ========")
            for i in range(n_warmup_runs):
                runner.run()
                print(runner.time)

            # Warmup
            print("======== Running ========")
            for i in range(n_runs):
                runner.run()
                print(runner.time)
                total += runner.time

            avg = total / n_runs
            print(f"======== {n_threads}, {test} Avg time: {avg} ========")
            out.write(f"{avg},")
            out.flush()

    out.close()


if __name__ == "__main__":
    main()



#csvs = []
#for inp in INPUTS:
#    for loop in LOOPS:
#        csv = ["{}/{}".format(inp, loop)]
#        for thr in THREADS:
#            cmd = "./bin/prefix_scan -o temp.txt -n {} -i tests/{} -l {}".format(
#                thr, inp, loop)
#            out = check_output(cmd, shell=True).decode("ascii")
#            m = re.search("time: (.*)", out)
#            if m is not None:
#                time = m.group(1)
#                csv.append(time)
#
#        csvs.append(csv)
#        sleep(0.5)
#
#header = ["microseconds"] + [str(x) for x in THREADS]
#
#print("\n")
#print(", ".join(header))
#for csv in csvs:
#    print (", ".join(csv))
