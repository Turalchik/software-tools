# ====MPI====
SRC_MPI = mpi/main.cpp
BIN_DIR_MPI = mpi/bin
TARGET_MPI = $(BIN_DIR_MPI)/main

NPROC = 6

build_mpi: $(BIN_DIR_MPI) $(TARGET_MPI)

$(BIN_DIR_MPI):
	mkdir -p $(BIN_DIR_MPI)

$(TARGET_MPI): $(SRC_MPI)
	mpic++ -g -Wall -o $(TARGET_MPI) $(SRC_MPI)

run_mpi: $(TARGET_MPI)
	mpiexec -n $(NPROC) $(TARGET_MPI)

clean_mpi:
	rm -rf $(BIN_DIR_MPI)

all_mpi: clean_mpi build_mpi run_mpi
# ===========

# ====OpenCL====
SRC_OPENCL = opencl/main.cpp
BIN_DIR_OPENCL = opencl/bin
TARGET_OPENCL = $(BIN_DIR_OPENCL)/main

build_opencl: $(BIN_DIR_OPENCL) $(TARGET_OPENCL)

$(BIN_DIR_OPENCL):
	mkdir -p $(BIN_DIR_OPENCL)

$(TARGET_OPENCL): $(SRC_OPENCL)
	g++ $(SRC_OPENCL) -lOpenCL -o $(TARGET_OPENCL)

run_opencl: $(TARGET_OPENCL)
	./$(TARGET_OPENCL)

clean_opencl:
	rm -rf $(BIN_DIR_OPENCL)

all_opencl: clean_opencl build_opencl run_opencl
# ==============

# ====OpenMP====
SRC_OPENMP = openmp/main.cpp
BIN_DIR_OPENMP = openmp/bin
TARGET_OPENMP = $(BIN_DIR_OPENMP)/main

build_openmp: $(BIN_DIR_OPENMP) $(TARGET_OPENMP)

$(BIN_DIR_OPENMP):
	mkdir -p $(BIN_DIR_OPENMP)

$(TARGET_OPENMP): $(SRC_OPENMP)
	g++ -fopenmp -o $(TARGET_OPENMP) $(SRC_OPENMP)

run_openmp: $(TARGET_OPENMP)
	./$(TARGET_OPENMP)

clean_openmp:
	rm -rf $(BIN_DIR_OPENMP)

all_openmp: clean_openmp build_openmp run_openmp
# ==============

clean:
	rm -rf $(BIN_DIR_MPI) $(BIN_DIR_OPENCL) $(BIN_DIR_OPENMP)