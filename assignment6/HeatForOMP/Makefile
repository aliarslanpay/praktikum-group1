# Compiler and flags
CC =  mpicc
CFLAGS = -O3 -fno-alias -xhost -qopenmp
LDFLAGS = -lm

# Targets and dependencies
TARGET = heat
OBJS = heat.o input.o misc.o timing.o relax_jacobi.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o : %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o $(TARGET) *~ *.ppm

remake: clean all

