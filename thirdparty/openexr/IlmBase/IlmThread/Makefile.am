## Process this file with automake to produce Makefile.in

lib_LTLIBRARIES = libIlmThread.la

libIlmThread_la_SOURCES = IlmThreadPool.h IlmThread.h \
			  IlmThreadSemaphore.h IlmThreadMutex.h \
			  IlmThreadPool.cpp IlmThread.cpp \
			  IlmThreadSemaphore.cpp IlmThreadMutex.cpp

if HAVE_PTHREAD
libIlmThread_la_SOURCES += IlmThreadPosix.cpp IlmThreadSemaphorePosix.cpp \
			   IlmThreadSemaphorePosixCompat.cpp \
			   IlmThreadMutexPosix.cpp
EXTRA_DIST_Thread = IlmThreadMutexWin32.cpp IlmThreadSemaphoreWin32.cpp \
		    IlmThreadWin32.cpp
else
libIlmThread_la_SOURCES += IlmThreadMutexWin32.cpp IlmThreadSemaphoreWin32.cpp \
			   IlmThreadWin32.cpp
EXTRA_DIST_Thread = IlmThreadPosix.cpp IlmThreadSemaphorePosix.cpp \
		    IlmThreadSemaphorePosixCompat.cpp \
		    IlmThreadMutexPosix.cpp
endif
libIlmThread_la_LDFLAGS = -version-info @LIBTOOL_VERSION@ -no-undefined
if LIB_SUFFIX_EXISTS
libIlmThread_la_LDFLAGS += -release @LIB_SUFFIX@
endif
libIlmThread_la_LIBADD = ../Iex/libIex.la

libIlmThreadincludedir = $(includedir)/OpenEXR

libIlmThreadinclude_HEADERS = IlmThreadPool.h IlmThread.h \
			      IlmThreadSemaphore.h IlmThreadMutex.h \
			      IlmThreadNamespace.h IlmThreadForward.h \
			      IlmThreadExport.h

noinst_HEADERS = 

EXTRA_DIST = $(noinst_HEADERS) $(EXTRA_DIST_Thread) CMakeLists.txt

INCLUDES = -I$(top_builddir) -I$(top_srcdir)/Iex -I$(top_srcdir)/config
