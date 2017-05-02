OBJS = main.cc layer.o weights.o network.o functions.o test.o cell.o weightsLSTM.o
CC = clang++
CFLAGS = -std=c++11 -Ofast -c
LFLAGS = -std=c++11 -Ofast

Build : $(OBJS)
	$(CC) $(LFLAGS) -o Build $(OBJS)

layer.o : layer.hh layer.cc weights.hh functions.hh
	$(CC) $(CFLAGS) layer.cc -o layer.o

weights.o : weights.cc weights.hh functions.hh
	$(CC) $(CFLAGS) weights.cc -o weights.o

network.o : network.hh network.cc layer.hh functions.hh
	$(CC) $(CFLAGS) network.cc -o network.o

functions.o : functions.hh functions.cc
	$(CC) $(CFLAGS) functions.cc -o functions.o

test.o : test.hh test.cc weights.hh network.hh
	$(CC) $(CFLAGS) test.cc -o test.o

cell.o : cell.hh cell.cc weightsLSTM.hh functions.hh
	$(CC) $(CFLAGS) cell.cc -o cell.o

weightsLSTM.o : weightsLSTM.hh weightsLSTM.cc functions.hh
	$(CC) $(CFLAGS) weightsLSTM.cc -o weightsLSTM.o
