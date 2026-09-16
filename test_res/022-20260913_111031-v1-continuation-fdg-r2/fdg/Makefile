ROOT := $(abspath $(CURDIR)/../../..)
NATIVE := $(ROOT)/native/hickit
CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -Wc++-compat -fno-fast-math -fno-unsafe-math-optimizations
LDFLAGS ?=
NATIVE_OBJS := $(NATIVE)/sdict.o $(NATIVE)/io.o $(NATIVE)/pair.o \
               $(NATIVE)/count.o $(NATIVE)/phase.o $(NATIVE)/bin.o \
               $(NATIVE)/fdg.o $(NATIVE)/image.o $(NATIVE)/view3d.o \
               $(NATIVE)/fdg_gpu_stub.o

.PHONY: all clean

all: build/fdg_bridge

build/fdg_bridge: fdg_bridge.c $(NATIVE_OBJS) | build
	$(CC) $(CFLAGS) -I$(NATIVE) fdg_bridge.c $(NATIVE_OBJS) \
		-o $@ $(LDFLAGS) -lm -lz

build:
	mkdir -p $@

clean:
	rm -f build/fdg_bridge
