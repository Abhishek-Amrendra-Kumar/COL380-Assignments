# COL380 Assignments

This repository contains my solutions for the COL380 course assignments (A1, A2 and A3). Each assignment is provided in its own folder with source files and minimal build notes below.

Prerequisites
- GNU C++ toolchain (g++ / clang++)
- GNU Make (for projects that include a Makefile)
- NVIDIA CUDA toolkit (for A2 if you want to build the CUDA solution)

Repository layout
- A1/: C++ implementation files for Assignment 1 (`functions.cpp`, `functions.h`).
- A2/: CUDA implementation and `makefile` for Assignment 2 (`main.cu`).
- A3/: C++ solution for Assignment 3 (`main.cpp`).

Build & run

- A1 (C++):
	- There is no top-level Makefile in this folder. If you have a `main.cpp` or a test harness, compile like:

		g++ -std=c++17 -O2 A1/*.cpp -I A1 -o a1_executable

	- If you use the course-provided tester/harness (outside this folder), place `functions.cpp`/`functions.h` into the tester folder and build using that tester's Makefile.

- A2 (CUDA):
	- A Makefile is provided in `A2/`. From the `A2` directory run:

		cd A2
		make

	- Alternatively compile directly with `nvcc`:

		nvcc -O2 A2/main.cu -o a2_executable

- A3 (C++):
	- Build with:

		g++ -std=c++17 -O2 A3/main.cpp -o A3/main

Notes
- For CUDA builds ensure your `nvcc` is on PATH and your system has a supported NVIDIA driver.
