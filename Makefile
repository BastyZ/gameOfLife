OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

main: opencl.c
	gcc -Wall -g opencl.c -o opencl $(OPTIONS)

clean:
	rm -rf opencl