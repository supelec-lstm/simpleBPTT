OBJS = main.cc layer.o weights.o network.o functions.o test.o cell.o weightsLSTM.o networkLSTM.o
CC = clang++
CFLAGS = -std=c++11 -Ofast -c
LFLAGS = -std=c++11 -Ofast

Build : $(OBJS)
	$(CC) $(LFLAGS) -o Build $(OBJS)

layer.o : neuronLayer/layer.hh neuronLayer/layer.cc neuronLayer/weights.hh functions.hh
	$(CC) $(CFLAGS) neuronLayer/layer.cc -o layer.o

weights.o : neuronLayer/weights.cc neuronLayer/weights.hh functions.hh
	$(CC) $(CFLAGS) neuronLayer/weights.cc -o weights.o

network.o : neuronLayer/network.hh neuronLayer/network.cc neuronLayer/layer.hh functions.hh
	$(CC) $(CFLAGS) neuronLayer/network.cc -o network.o

functions.o : functions.hh functions.cc
	$(CC) $(CFLAGS) functions.cc -o functions.o

cell.o : lstmCell/cell.hh lstmCell/cell.cc lstmCell/weightsLSTM.hh functions.hh
	$(CC) $(CFLAGS) lstmCell/cell.cc -o cell.o

weightsLSTM.o : lstmCell/weightsLSTM.hh lstmCell/weightsLSTM.cc functions.hh
	$(CC) $(CFLAGS) lstmCell/weightsLSTM.cc -o weightsLSTM.o

networkLSTM.o : lstmCell/networkLSTM.cc lstmCell/networkLSTM.hh lstmCell/cell.hh functions.hh lstmCell/weightsLSTM.hh
	$(CC) $(CFLAGS) lstmCell/networkLSTM.cc -o networkLSTM.o

test.o : test.hh test.cc neuronLayer/weights.hh neuronLayer/network.hh lstmCell/weightsLSTM.hh lstmCell/networkLSTM.hh
	$(CC) $(CFLAGS) test.cc -o test.o
