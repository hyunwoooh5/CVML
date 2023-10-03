#CXX=g++
#CXXFLAGS += -O3 -I${HOME}/local/include -std=c++11 -I. -DNDEBUG -g 
#CXXFLAGS += -O3 -I${HOME}/Programs/include -std=c++11 -I. -DNDEBUG  

#CXXFLAGS += -O3 -I${HOME}/local/include -std=c++11 -DNDEBUG
CXXFLAGS += -O3 -I${HOME}/Programs/include -std=c++11 -I. -DNDEBUG 

CXXFLAGS += -O3 -I${HOME}/local/include -I/home/oh/Downloads/eigen-3.3.9/Eigen/Dense -std=c++11 -DNDEBUG
LDLIBS = -lm


#is for comments
