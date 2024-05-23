CC =  mpicc
CFLAGS = -O


all: reduction 

reduction : reduction.o
	$(CC) $(CFLAGS) -o reduction $+ -lm  

%.o : %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o reduction *~ *.ppm

remake : clean all
