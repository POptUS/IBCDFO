#
#   TRON main directory
#

L_ARCH   = $(ARCH)
LIB_NAME = d-$(L_ARCH).a
LIB_NAME = d-linux.a

#@echo $(LIB_NAME)
OPTFLAGS = -O -fPIC

CFLAGS    = $(OPTFLAGS) 
FFLAGS    = $(OPTFLAGS)
# CPPFLAGS  = -DDEBUG -DMATLAB
CPPFLAGS  = -DDDEBUG

# Libraries.

TRON   = /home/more/minpack2/tron/src/$(LIB_NAME)
BLAS   = /home/more/minpack2/blas/double/$(LIB_NAME)
LAPACK = /home/more/minpack2/lapack/double/$(LIB_NAME)
NMTR  = /home/more/minpack2/nmtr/src/$(LIB_NAME)
TPROBS = /home/more/minpack2/tprobs/$(LIB_NAME)
ACML = /home/wild/software/acml4.1.0/gfortran64/lib/libacml.a

LIBS = -llapack -lblas


#gfortran -m64 driver.f -L/opt/acml4.1.0/gfortran64/lib -static -lacml
#gfortran -m64 driver.f /opt/acml4.1.0/gfortran64/lib/libacml.a

CFLAGS = -O3 

FFLAGS = -O -fPIC

gqt: gqt.c dgqt.o destsv.o
	mex CFLAGS="-O3 -fPIC" $(LIBS) gqt.c dgqt.o destsv.o 

.SUFFIXES: .f .o .c .h

.f.o: $*.f
	$(FC)  -c $(FFLAGS) $*.f

.c.o: $*.c
	cc -c $*.c
