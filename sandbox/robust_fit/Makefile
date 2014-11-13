CC = g++

LIBARMA = -L$(HOME)/lib/armadillo/lib64 -larmadillo
INCARMA = -I$(HOME)/lib/armadillo/include -I/net/usr/local/boost_1_48_0

CFLAGS = -lm -lpthread $(LIBARMA) $(INCARMA)

SOURCE = *.cpp

MYPROG = test_2

all: $(MYPROG)

$(MYPROG): $(SOURCE)
	    $(CC) $(SOURCE) -o $(MYPROG) $(CFLAGS)

clean:
	    rm -f $(MYPROG)
