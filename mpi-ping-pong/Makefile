CC = mpicc
TARGET = ping_pong

SRC = ping_pong.c
LIBS = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)
